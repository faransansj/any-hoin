import pytest
import numpy as np
import torch
from pathlib import Path
import shutil
from PIL import Image
from unittest.mock import MagicMock, patch

from dataset import HoloDataset, get_train_transforms, get_val_transforms
from crawling.danbooru_crawler import DanbooruCrawler
from fastapi import HTTPException
from studio.routers.images import _resolve_image_id, _resolve_label_dir
from studio.routers.labels import _guard_name
import studio.characters as characters_store
import studio.routers.characters as characters_router
import studio.routers.crawl as crawl_router
from scripts.sync_backend import sync_command

# ──────────────────────────────────────────────
# Dataset Tests
# ──────────────────────────────────────────────


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """테스트용 가상 데이터셋 구조 생성"""
    classes = ["char_a", "char_b"]
    for cls in classes:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(10):
            img = Image.new("RGB", (100, 100), color="red")
            img.save(cls_dir / f"img_{i}.jpg")
    return tmp_path


def test_holo_dataset_init(mock_dataset_dir):
    """클래스 자동 생성 및 샘플 수집 검증"""
    ds = HoloDataset(mock_dataset_dir, split="train", val_ratio=0, test_ratio=0)
    assert len(ds.classes) == 2
    assert "char_a" in ds.classes
    assert "char_b" in ds.classes
    assert len(ds.samples) == 20


def test_holo_dataset_reads_nested_others(tmp_path):
    """others/<class>에 격리된 이미지도 others 클래스로 학습에 포함."""
    nested_dir = tmp_path / "others" / "char_c"
    nested_dir.mkdir(parents=True)
    for i in range(3):
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(nested_dir / f"img_{i}.jpg")

    ds = HoloDataset(tmp_path, split="train", val_ratio=0, test_ratio=0)

    assert ds.classes == ["others"]
    assert len(ds.samples) == 3


def test_holo_dataset_skips_others_subdir_when_class_exists(tmp_path):
    """top-level class가 있으면 others/<class>는 중복 학습 라벨에서 제외."""
    cls_dir = tmp_path / "char_c"
    nested_dir = tmp_path / "others" / "char_c"
    cls_dir.mkdir()
    nested_dir.mkdir(parents=True)

    Image.new("RGB", (100, 100), color="red").save(cls_dir / "current.jpg")
    Image.new("RGB", (100, 100), color="blue").save(nested_dir / "old.jpg")

    ds = HoloDataset(tmp_path, split="train", val_ratio=0, test_ratio=0)

    assert ds.classes == ["char_c"]
    assert len(ds.samples) == 1
    assert Path(ds.samples[0][0]).name == "current.jpg"


def test_build_dataloaders_rejects_empty_dataset(tmp_path):
    from dataset import build_dataloaders

    with pytest.raises(RuntimeError, match="No training images"):
        build_dataloaders(tmp_path, num_workers=0)


def test_holo_dataset_split(mock_dataset_dir):
    """데이터 분할 비율 검증"""
    # 20장 중 10% val(2장), 10% test(2장) -> train 16장
    train_ds = HoloDataset(
        mock_dataset_dir, split="train", val_ratio=0.1, test_ratio=0.1
    )
    val_ds = HoloDataset(mock_dataset_dir, split="val", val_ratio=0.1, test_ratio=0.1)
    test_ds = HoloDataset(mock_dataset_dir, split="test", val_ratio=0.1, test_ratio=0.1)

    assert len(train_ds) == 16
    assert len(val_ds) == 2
    assert len(test_ds) == 2

    train_paths = {p for p, _ in train_ds.samples}
    val_paths = {p for p, _ in val_ds.samples}
    test_paths = {p for p, _ in test_ds.samples}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


def test_transforms_output_shape():
    """트랜스폼 결과 텐서 크기 검증 (224x224)"""
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)

    train_res = train_tf(image=img)["image"]
    val_res = val_tf(image=img)["image"]

    assert train_res.shape == (3, 224, 224)
    assert val_res.shape == (3, 224, 224)


def test_image_path_guards_reject_traversal():
    """이미지 API 경로 헬퍼가 dataset 밖 접근을 차단해야 함."""
    with pytest.raises(HTTPException):
        _resolve_image_id("../outside.jpg")
    with pytest.raises(HTTPException):
        _resolve_label_dir("../outside")
    with pytest.raises(HTTPException):
        _resolve_label_dir("parent/child")


def test_label_guard_rejects_unsafe_names():
    """라벨 CRUD가 dataset 밖 경로를 만들거나 지우지 못해야 함."""
    for name in ("..", ".hidden", "a/b", "a\\b", "a.b"):
        with pytest.raises(HTTPException):
            _guard_name(name)


def test_character_discovery_recovers_existing_dataset(tmp_path, monkeypatch):
    """세션 재시작 후 characters.json이 없어도 dataset/raw 폴더를 다시 등록."""
    data_dir = tmp_path / "dataset" / "raw"
    char_dir = data_dir / "hina_(blue_archive)"
    char_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color="red").save(char_dir / "img.jpg")

    chars_file = tmp_path / "characters.json"
    monkeypatch.setattr(characters_store, "CHARACTERS_FILE", chars_file)
    monkeypatch.setattr(characters_router, "DATASET_DIR", data_dir)

    discovered = characters_router.discover_dataset_characters()
    assert discovered["missing"][0]["key"] == "hina_(blue_archive)"

    recovered = characters_router.recover_dataset_characters({"keys": ["hina_(blue_archive)"]})
    assert recovered["imported"] == 1
    assert characters_store.load()["hina_(blue_archive)"]["tag"] == "hina_(blue_archive)"


def test_danbooru_tag_search_uses_autocomplete_alias(monkeypatch):
    """실제 캐릭터 alias 입력도 autocomplete 결과로 Danbooru 태그를 찾는다."""
    calls = []

    def fake_get(url, params, **_kwargs):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url == crawl_router.DANBOORU_AUTOCOMPLETE_URL:
            calls.append(("autocomplete", params["search[query]"]))
            response.json.return_value = [{
                "label": "hina (blue archive)",
                "value": "hina_(blue_archive)",
                "category": 4,
                "post_count": 17740,
                "antecedent": "sorasaki_hina",
                "tag": {"is_deprecated": False, "post_count": 17740},
            }]
        else:
            calls.append(("other", params))
            response.json.return_value = []
        return response

    monkeypatch.setattr(crawl_router._req, "get", fake_get)

    result = crawl_router.search_tags("Sorasaki Hina", limit=10)

    assert calls[:2] == [("autocomplete", "Sorasaki Hina"), ("autocomplete", "sorasaki_hina")]
    assert result["tags"][0]["name"] == "hina_(blue_archive)"
    assert result["tags"][0]["antecedent"] == "sorasaki_hina"
    assert result["tags"][0]["post_count"] == 17740


def test_danbooru_tag_search_route_registered():
    """FastAPI 라우터에 태그 검색 엔드포인트가 GET으로 등록되어야 함."""
    assert any(
        route.path == "/crawl/tags/search" and "GET" in route.methods
        for route in crawl_router.router.routes
    )


def test_danbooru_tag_search_normalizes_and_falls_back(monkeypatch):
    """공백 검색어를 Danbooru 태그 패턴으로 바꾸고 contains fallback까지 수행."""
    tag_patterns = []

    def fake_get(url, params, **_kwargs):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url == crawl_router.DANBOORU_AUTOCOMPLETE_URL:
            response.json.return_value = []
        elif url == crawl_router.DANBOORU_TAGS_URL:
            tag_patterns.append(params["search[name_matches]"])
            response.json.return_value = (
                [{
                    "name": "hina_(blue_archive)",
                    "post_count": 17740,
                    "category": 4,
                    "is_deprecated": False,
                }]
                if tag_patterns[-1] == "*hina*blue*archive*"
                else []
            )
        else:
            response.json.return_value = []
        return response

    monkeypatch.setattr(crawl_router._req, "get", fake_get)

    result = crawl_router.search_tags("Hina Blue Archive", limit=10)

    assert tag_patterns == ["hina_blue_archive*", "*hina_blue_archive*", "*hina*blue*archive*"]
    assert result["tags"][0]["name"] == "hina_(blue_archive)"
    assert result["tags"][0]["post_count"] == 17740


def test_backend_sync_commands_are_unified_profiles():
    assert sync_command("cpu") == ["uv", "sync"]
    assert sync_command("mps") == ["uv", "sync"]
    assert sync_command("cuda") == ["uv", "sync", "--extra", "cuda"]
    assert sync_command("arc") == ["uv", "sync", "--extra", "arc"]
    assert sync_command("rocm") == ["uv", "sync", "--extra", "rocm"]


# ──────────────────────────────────────────────
# Crawler Tests
# ──────────────────────────────────────────────


@patch("requests.Session.get")
def test_crawler_get_posts(mock_get, tmp_path):
    """API 응답 파싱 로직 검증"""
    # 가짜 API 응답 설정
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 1, "file_url": "http://test.com/1.jpg", "rating": "g"},
        {
            "id": 2,
            "file_url": "http://test.com/2.jpg",
            "rating": "x",
        },  # SFW 필터링 대상
    ]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    crawler = DanbooruCrawler(output_dir=tmp_path)
    posts = crawler._get_posts("test_tag", 1)

    assert len(posts) == 2
    assert posts[0]["id"] == 1


@patch("crawling.danbooru_crawler.time.sleep", return_value=None)
@patch("requests.Session.get")
def test_crawler_collect_queue(mock_get, _mock_sleep, tmp_path):
    """큐 수집 시 SFW 필터링 및 확장자 검증"""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 1, "file_url": "http://test.com/1.jpg", "rating": "g"},  # OK
        {"id": 2, "file_url": "http://test.com/2.png", "rating": "s"},  # OK
        {"id": 3, "file_url": "http://test.com/3.txt", "rating": "g"},  # Invalid Ext
        {"id": 4, "file_url": "http://test.com/4.jpg", "rating": "x"},  # Not SFW
    ]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    crawler = DanbooruCrawler(output_dir=tmp_path)
    char_dir = tmp_path / "test_char"
    char_dir.mkdir()

    queue = crawler._collect_queue("test_tag", char_dir, 10)

    # 1, 2번만 통과해야 함
    assert len(queue) == 2
    assert "1.jpg" in queue[0][1].name
    assert "2.png" in queue[1][1].name


@patch("crawling.danbooru_crawler.time.sleep", return_value=None)
def test_crawler_collect_queue_honors_large_need(_mock_sleep, tmp_path):
    """max_images > 1000일 때 10페이지 하드캡 없이 후보를 수집해야 함"""
    crawler = DanbooruCrawler(output_dir=tmp_path)
    char_dir = tmp_path / "test_char"
    char_dir.mkdir()

    def fake_posts(_tag, page):
        start = (page - 1) * 100
        return [
            {
                "id": start + i,
                "file_url": f"http://test.com/{start + i}.jpg",
                "rating": "g",
            }
            for i in range(100)
        ]

    with patch.object(crawler, "_get_posts", side_effect=fake_posts) as mock_get_posts:
        queue = crawler._collect_queue("test_tag", char_dir, 1200)

    assert len(queue) == 1200
    assert mock_get_posts.call_count == 12
