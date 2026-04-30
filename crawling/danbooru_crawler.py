"""
Danbooru Crawler for Hololive Character Classification
- SFW only (rating:general,sensitive)
- MD5 중복 제거
- 병렬 이미지 다운로드 (--workers)
- Pillow 이미지 유효성 검사
- .env 자격증명 자동 로드
- Rich CLI UI
"""

import io
import json
import os
import time
import hashlib
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from PIL import Image, ImageOps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)
from rich import box

console = Console()

# ──────────────────────────────────────────────
# 홀로라이브 전체 멤버 태그 목록 (나무위키 문서 존재 기준)
# danbooru 태그 형식: 캐릭터명_(hololive)
# ──────────────────────────────────────────────
HOLOLIVE_MEMBERS = {
    # JP 0기생
    "tokino_sora":        "tokino_sora_(hololive)",
    "roboco":             "roboco-san",
    "sakura_miko":        "sakura_miko",
    "hoshimachi_suisei":  "hoshimachi_suisei",
    "azki":               "azki_(hololive)",

    # JP 1기생
    "yozora_mel":         "yozora_mel",
    "shirakami_fubuki":   "shirakami_fubuki",
    "natsuiro_matsuri":   "natsuiro_matsuri",
    "aki_rosenthal":      "aki_rosenthal",
    "akai_haato":         "akai_haato",

    # JP 2기생
    "minato_aqua":        "minato_aqua",
    "murasaki_shion":     "murasaki_shion",
    "nakiri_ayame":       "nakiri_ayame",
    "yuzuki_choco":       "yuzuki_choco",
    "oozora_subaru":      "oozora_subaru",

    # JP 3기생 (Fantasy)
    "usada_pekora":       "usada_pekora",
    "shiranui_flare":     "shiranui_flare",
    "shirogane_noel":     "shirogane_noel",
    "houshou_marine":     "houshou_marine",

    # JP 4기생
    "amane_kanata":       "amane_kanata",
    "tsunomaki_watame":   "tsunomaki_watame",
    "tokoyami_towa":      "tokoyami_towa",
    "himemori_luna":      "himemori_luna",

    # JP 5기생 (Nepolabo)
    "yukihana_lamy":      "yukihana_lamy",
    "momosuzu_nene":      "momosuzu_nene",
    "shishiro_botan":     "shishiro_botan",
    "omaru_polka":        "omaru_polka",

    # JP 6기생 (Secret Base)
    "laplus_darknesss":   "la+_darknesss",
    "takane_lui":         "takane_lui",
    "hakui_koyori":       "hakui_koyori",
    "sakamata_chloe":     "sakamata_chloe",
    "kazama_iroha":       "kazama_iroha",

    # JP 7기생
    "hiodoshi_ao":        "hiodoshi_ao",
    "otonose_kanade":     "otonose_kanade",
    "ichijou_ririka":     "ichijou_ririka",
    "juufuutei_raden":    "juufuutei_raden",
    "todoroki_hajime":    "todoroki_hajime",

    # EN Myth
    "mori_calliope":      "mori_calliope",
    "takanashi_kiara":    "takanashi_kiara",
    "ninomae_inanis":     "ninomae_ina'nis",
    "gawr_gura":          "gawr_gura",
    "watson_amelia":      "watson_amelia",

    # EN Council / Promise
    "ceres_fauna":        "ceres_fauna",
    "ouro_kronii":        "ouro_kronii",
    "nanashi_mumei":      "nanashi_mumei",
    "hakos_baelz":        "hakos_baelz",

    # EN Advent
    "shiori_novella":     "shiori_novella",
    "koseki_bijou":       "koseki_bijou",
    "nerissa_ravencroft": "nerissa_ravencroft",
    "fuwamoco":           "fuwawa_abyssgard",

    # EN Justice
    "elizabeth_rose_bloodflame": "elizabeth_rose_bloodflame",
    "gigi_murin":         "gigi_murin",
    "cecilia_immergreen": "cecilia_immergreen",
    "raora_panthera":     "raora_panthera",

    # ID Gen1
    "ayunda_risu":        "ayunda_risu",
    "moona_hoshinova":    "moona_hoshinova",
    "airani_iofifteen":   "airani_iofifteen",

    # ID Gen2
    "kureiji_ollie":      "kureiji_ollie",
    "anya_melfissa":      "anya_melfissa",
    "pavolia_reine":      "pavolia_reine",

    # ID Gen3
    "vestia_zeta":        "vestia_zeta",
    "kaela_kovalskia":    "kaela_kovalskia",
    "kobo_kanaeru":       "kobo_kanaeru",

    # DEV_IS ReGLOSS
    "hifumi_ichimatsu":   "ichimatsu_hifumi",
    "isaki_riona":        "isaki_riona",
    "mizumiya_su":        "mizumiya_su",
    "rindo_chihaya":      "rindo_chihaya",
}

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
BASE_URL      = "https://danbooru.donmai.us"
ALLOWED_RATINGS = {"g", "s"}   # general, sensitive (SFW)
ALLOWED_EXT     = {".jpg", ".jpeg", ".png", ".webp"}
POSTS_PER_PAGE  = 100
QUEUE_PAGE_FACTOR = 3           # 필터/중복/다운로드 실패 여유분
MIN_QUEUE_PAGES = 10
PAGE_SLEEP      = 0.5           # API 페이지 요청 간격 (초)
CRAWL_EVENT_PREFIX = "__HOLOSCOPE_CRAWL_EVENT__ "
MB = 1024 * 1024
DEFAULT_RESIZE_THRESHOLD_MB = 4.0
DEFAULT_RESIZE_MAX_SIDE = 1536
DEFAULT_RESIZE_QUALITY = 88


class DanbooruCrawler:
    def __init__(
        self,
        username: str = "",
        api_key: str = "",
        output_dir: Path = Path("./dataset/raw"),
        workers: int = 4,
        emit_events: bool = False,
        resize_large_images: bool = False,
        resize_threshold_mb: float = DEFAULT_RESIZE_THRESHOLD_MB,
        resize_max_side: int = DEFAULT_RESIZE_MAX_SIDE,
        resize_quality: int = DEFAULT_RESIZE_QUALITY,
    ):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "HoloScope-Crawler/1.0"
        self.auth = (username, api_key) if username else None
        self.output_dir = Path(output_dir)
        self.workers = workers
        self.emit_events = emit_events
        self.resize_large_images = resize_large_images
        self.resize_threshold_bytes = int(max(resize_threshold_mb, 0.0) * MB)
        self.resize_max_side = max(512, int(resize_max_side))
        self.resize_quality = max(50, min(100, int(resize_quality)))
        self._hash_lock = Lock()
        self._health_lock = Lock()
        self.seen_hashes: set[str] = set()
        self._last_progress_emit = 0.0
        self._run_context: dict = {}
        self.health = {
            "account_mode": "authenticated" if self.auth else "anonymous",
            "api_requests": 0,
            "api_errors": 0,
            "download_errors": 0,
            "rate_limited": False,
            "last_status": None,
            "last_error": None,
            "retry_after": None,
            "last_api_at": None,
            "last_latency_ms": None,
            "resized_images": 0,
            "resize_saved_bytes": 0,
        }

    def _emit_event(self, event: str, payload: dict) -> None:
        if not self.emit_events:
            return
        print(
            CRAWL_EVENT_PREFIX + json.dumps({"event": event, **payload}, ensure_ascii=False),
            flush=True,
        )

    def _health_snapshot(self) -> dict:
        return dict(self.health)

    def _eta(self, remaining: int, speed: float) -> int:
        if remaining <= 0:
            return 0
        if speed <= 0:
            return -1
        return int(remaining / speed)

    def _emit_progress(
        self,
        *,
        char_name: str,
        char_index: int,
        phase: str,
        collected: int,
        target: int,
        char_started_at: float,
        queue_size: int = 0,
        force: bool = False,
        status: str | None = None,
    ) -> None:
        if not self.emit_events:
            return
        now = time.monotonic()
        if not force and now - self._last_progress_emit < 1.0:
            return
        self._last_progress_emit = now

        ctx = self._run_context
        started_at = float(ctx.get("started_at") or now)
        completed_images = int(ctx.get("completed_images") or 0)
        total_target = int(ctx.get("total_target") or target)
        total_downloaded = min(completed_images + collected, total_target)
        elapsed = max(now - started_at, 0.001)
        speed = total_downloaded / elapsed if total_downloaded > 0 else 0.0

        char_elapsed = max(now - char_started_at, 0.001)
        char_speed = max(collected - int(ctx.get("current_existing") or 0), 0) / char_elapsed
        char_remaining = max(target - collected, 0)
        total_remaining = max(total_target - total_downloaded, 0)

        self._emit_event("progress", {
            "current_character": char_name,
            "current_index": char_index,
            "total_characters": int(ctx.get("total_characters") or 0),
            "phase": phase,
            "status": status,
            "char_downloaded": collected,
            "char_target": target,
            "char_queue": queue_size,
            "char_pct": round((collected / target) * 100, 2) if target > 0 else 0,
            "char_eta_sec": self._eta(char_remaining, char_speed),
            "completed_characters": int(ctx.get("completed_characters") or 0),
            "included": int(ctx.get("included") or 0),
            "skipped": int(ctx.get("skipped") or 0),
            "below_threshold": int(ctx.get("below_threshold") or 0),
            "total_downloaded": total_downloaded,
            "total_target": total_target,
            "total_pct": round((total_downloaded / total_target) * 100, 2) if total_target > 0 else 0,
            "overall_eta_sec": self._eta(total_remaining, speed),
            "speed_img_s": round(speed, 3),
            "health": self._health_snapshot(),
            "updated_at": time.time(),
        })

    # ── API ──────────────────────────────────

    def _get_posts(self, tag: str, page: int | str) -> list[dict]:
        params = {
            "tags":  f"{tag} rating:general,sensitive",
            "limit": POSTS_PER_PAGE,
            "page":  page,
        }
        started = time.monotonic()
        self.health["api_requests"] += 1
        try:
            r = self.session.get(
                f"{BASE_URL}/posts.json",
                params=params,
                auth=self.auth,
                timeout=15,
            )
            self.health["last_status"] = r.status_code
            self.health["last_api_at"] = time.time()
            self.health["last_latency_ms"] = int((time.monotonic() - started) * 1000)
            self.health["retry_after"] = r.headers.get("Retry-After")
            self.health["rate_limited"] = r.status_code == 429
            r.raise_for_status()
            self.health["last_error"] = None
            return r.json()
        except Exception as e:
            self.health["api_errors"] += 1
            self.health["last_error"] = str(e)
            console.print(f"  [red]API 오류:[/] {e}")
            return []

    # ── 다운로드 ─────────────────────────────

    def _record_resize_savings(self, saved_bytes: int) -> None:
        if saved_bytes <= 0:
            return
        with self._health_lock:
            self.health["resized_images"] += 1
            self.health["resize_saved_bytes"] += saved_bytes

    def _maybe_resize_content(self, content: bytes, save_path: Path) -> bytes:
        if not self.resize_large_images:
            return content
        if len(content) < self.resize_threshold_bytes:
            return content

        try:
            with Image.open(io.BytesIO(content)) as src:
                if getattr(src, "is_animated", False):
                    return content

                image = ImageOps.exif_transpose(src)
                original_format = (src.format or save_path.suffix.lstrip(".") or "JPEG").upper()
                if original_format == "JPG":
                    original_format = "JPEG"
                if original_format not in {"JPEG", "PNG", "WEBP"}:
                    return content

                if max(image.width, image.height) > self.resize_max_side:
                    image.thumbnail((self.resize_max_side, self.resize_max_side), Image.LANCZOS)

                save_kwargs = {"optimize": True}
                if original_format in {"JPEG", "WEBP"}:
                    if image.mode not in {"RGB", "L"}:
                        image = image.convert("RGB")
                    save_kwargs["quality"] = self.resize_quality
                    if original_format == "JPEG":
                        save_kwargs["progressive"] = True
                elif original_format == "PNG":
                    save_kwargs["compress_level"] = 9

                out = io.BytesIO()
                image.save(out, original_format, **save_kwargs)
                resized = out.getvalue()
        except Exception:
            return content

        if resized and len(resized) < len(content):
            self._record_resize_savings(len(content) - len(resized))
            return resized
        return content

    def _download_one(self, url: str, save_path: Path) -> str:
        """단일 이미지 다운로드. 반환값: 'ok' | 'dup' | 'invalid' | 'error'"""
        try:
            r = self.session.get(url, timeout=20)
            if r.status_code == 429:
                with self._health_lock:
                    self.health["rate_limited"] = True
                    self.health["last_status"] = 429
                    self.health["retry_after"] = r.headers.get("Retry-After")
                    self.health["download_errors"] += 1
                return "error"
            r.raise_for_status()

            # MD5 중복 체크 (세션 내)
            img_hash = hashlib.md5(r.content).hexdigest()
            with self._hash_lock:
                if img_hash in self.seen_hashes:
                    return "dup"
                self.seen_hashes.add(img_hash)

            # Pillow 유효성 검사
            try:
                Image.open(io.BytesIO(r.content)).verify()
            except Exception:
                return "invalid"

            save_path.write_bytes(self._maybe_resize_content(r.content, save_path))
            return "ok"
        except Exception:
            with self._health_lock:
                self.health["download_errors"] += 1
            return "error"

    def _collect_queue(
        self,
        tag: str,
        char_dir: Path,
        need: int,
        *,
        char_name: str = "",
        char_index: int = 1,
        collected: int = 0,
        target: int | None = None,
        char_started_at: float | None = None,
    ) -> list[tuple[str, Path]]:
        """다운로드할 (url, 저장경로) 목록을 API에서 수집 (순차적으로 rate-limit 준수)"""
        queue: list[tuple[str, Path]] = []
        seen_paths: set[Path] = set()
        page: int | str = 1
        last_page_min_id: int | None = None
        target = target if target is not None else need + collected
        char_started_at = char_started_at if char_started_at is not None else time.monotonic()
        # Danbooru caps sequential pagination at ~20 pages for anonymous/basic accounts.
        # max_seq_pages is the threshold to proactively switch to cursor (page=b{id}) mode.
        max_seq_pages = max(
            MIN_QUEUE_PAGES,
            ((need + POSTS_PER_PAGE - 1) // POSTS_PER_PAGE) * QUEUE_PAGE_FACTOR,
        )

        while len(queue) < need:
            posts = self._get_posts(tag, page)
            if not posts:
                # Sequential pages ran dry — switch to cursor mode if we have a reference ID.
                if isinstance(page, int) and last_page_min_id is not None:
                    page = f"b{last_page_min_id}"
                    last_page_min_id = None
                    continue
                break

            for post in posts:
                if len(queue) >= need:
                    break

                file_url = post.get("file_url") or post.get("large_file_url")
                if not file_url:
                    continue

                ext = Path(file_url).suffix.lower()
                if ext not in ALLOWED_EXT:
                    continue

                if post.get("rating", "e") not in ALLOWED_RATINGS:
                    continue

                save_path = char_dir / f"{post['id']}{ext}"
                if save_path in seen_paths or save_path.exists():
                    continue

                seen_paths.add(save_path)
                queue.append((file_url, save_path))

            # Advance pagination: track min post ID for cursor-based continuation.
            page_ids = [p["id"] for p in posts if isinstance(p.get("id"), int)]
            if page_ids:
                batch_min_id = min(page_ids)
                if isinstance(page, int):
                    last_page_min_id = batch_min_id
                else:
                    new_cursor = f"b{batch_min_id}"
                    if new_cursor == page:
                        break  # No progress in cursor mode
                    page = new_cursor
            else:
                break  # No usable post IDs — can't advance pagination

            if isinstance(page, int):
                page += 1
                # Proactively switch to cursor when sequential limit is reached.
                if page > max_seq_pages:
                    page = f"b{last_page_min_id}"
                    last_page_min_id = None

            self._emit_progress(
                char_name=char_name,
                char_index=char_index,
                phase="collecting",
                collected=collected,
                target=target,
                queue_size=len(queue),
                char_started_at=char_started_at,
            )
            if len(queue) < need:
                time.sleep(PAGE_SLEEP)

        return queue

    # ── 캐릭터 크롤링 ────────────────────────

    def crawl_character(
        self,
        char_name: str,
        tag: str,
        max_images: int,
        min_images: int,
        progress: Progress,
        task_id,
        char_index: int,
        total_chars: int,
    ) -> tuple[str, int]:
        """(status, count) 반환.  status: 'included' | 'skipped' | 'below_threshold'"""
        char_dir = self.output_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        self._restore_from_others(char_name)

        existing = len([
            f for f in char_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXT
        ])
        char_started_at = time.monotonic()
        self._run_context["current_existing"] = existing
        self._emit_progress(
            char_name=char_name,
            char_index=char_index,
            phase="starting",
            collected=existing,
            target=max_images,
            char_started_at=char_started_at,
            force=True,
        )

        # 이미 충분히 수집된 경우
        if existing >= max_images:
            progress.update(
                task_id,
                completed=max_images,
                total=max_images,
                description=f"[dim]{char_name:<26}[/]",
            )
            self._emit_progress(
                char_name=char_name,
                char_index=char_index,
                phase="done",
                status="skipped",
                collected=existing,
                target=max_images,
                char_started_at=char_started_at,
                force=True,
            )
            return "skipped", existing

        collected = existing
        progress.update(task_id, completed=collected, total=max_images)

        # 필요한 이미지 목록 수집 (API 순차 호출)
        queue = self._collect_queue(
            tag,
            char_dir,
            max_images - collected,
            char_name=char_name,
            char_index=char_index,
            collected=collected,
            target=max_images,
            char_started_at=char_started_at,
        )
        self._emit_progress(
            char_name=char_name,
            char_index=char_index,
            phase="downloading",
            collected=collected,
            target=max_images,
            queue_size=len(queue),
            char_started_at=char_started_at,
            force=True,
        )

        # 병렬 다운로드
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._download_one, url, path): path
                for url, path in queue
            }
            for future in as_completed(futures):
                if future.result() == "ok":
                    collected += 1
                    progress.update(task_id, completed=min(collected, max_images))
                    self._emit_progress(
                        char_name=char_name,
                        char_index=char_index,
                        phase="downloading",
                        collected=collected,
                        target=max_images,
                        queue_size=len(queue),
                        char_started_at=char_started_at,
                    )

        if collected < min_images:
            return "below_threshold", collected
        return "included", collected

    def _unique_path(self, path: Path) -> Path:
        """Return a non-existing sibling path without overwriting existing files."""
        if not path.exists():
            return path
        idx = 1
        while True:
            candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
            if not candidate.exists():
                return candidate
            idx += 1

    def _same_file_contents(self, a: Path, b: Path) -> bool:
        if not a.is_file() or not b.is_file():
            return False
        if a.stat().st_size != b.stat().st_size:
            return False
        return hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()

    def _move_file(self, src: Path, dst: Path) -> int:
        if dst.exists():
            if self._same_file_contents(src, dst):
                src.unlink()
                return 0
            dst = self._unique_path(dst)
        shutil.move(str(src), str(dst))
        return 1

    def _merge_dir_contents(self, src: Path, dst: Path) -> int:
        """Merge src directory contents into dst, then remove the empty src."""
        if not src.exists():
            return 0
        if src.resolve() == dst.resolve():
            return 0

        dst.mkdir(parents=True, exist_ok=True)
        moved = 0
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                moved += self._merge_dir_contents(item, target)
                continue

            moved += self._move_file(item, target)

        try:
            src.rmdir()
        except OSError:
            pass
        return moved

    def _move_to_others(self, char_name: str) -> int:
        """Move a below-threshold class into others without nested duplicate dirs."""
        src = self.output_dir / char_name
        dst = self.output_dir / "others" / char_name

        # Repair legacy nested dirs produced by shutil.move(src, existing_dst).
        legacy_nested = dst / char_name
        moved = self._merge_dir_contents(legacy_nested, dst)
        moved += self._merge_dir_contents(src, dst)
        return moved

    def _restore_from_others(self, char_name: str) -> int:
        """Bring previously below-threshold images back when recrawling a class."""
        char_dir = self.output_dir / char_name
        other_dir = self.output_dir / "others" / char_name

        # Flatten any legacy others/<char>/<char> directory first.
        legacy_nested = other_dir / char_name
        moved = self._merge_dir_contents(legacy_nested, char_dir)
        moved += self._merge_dir_contents(other_dir, char_dir)
        return moved

    # ── 전체 실행 ────────────────────────────

    def run(
        self,
        min_images: int = 500,
        max_images: int = 1000,
        members: dict | None = None,
    ) -> dict:
        """members가 None이면 HOLOLIVE_MEMBERS 전체를 크롤."""
        target = members if members is not None else HOLOLIVE_MEMBERS
        self.output_dir.mkdir(parents=True, exist_ok=True)
        total_chars = len(target)
        total_target = total_chars * max_images
        self._run_context = {
            "started_at": time.monotonic(),
            "total_characters": total_chars,
            "total_target": total_target,
            "completed_characters": 0,
            "completed_images": 0,
            "included": 0,
            "skipped": 0,
            "below_threshold": 0,
            "current_existing": 0,
        }

        auth_label = (
            f"[cyan]{self.auth[0]}[/]" if self.auth else "[yellow]익명[/yellow] (rate limit 있음)"
        )
        console.print(
            Panel.fit(
                f"[bold white]HoloScope Danbooru Crawler[/]\n"
                f"계정 [dim]│[/] {auth_label}\n"
                f"대상 [dim]│[/] [cyan]{len(target)}명[/]  "
                f"기준 [dim]│[/] [green]{min_images}[/]–[green]{max_images}[/]장  "
                f"워커 [dim]│[/] [cyan]{self.workers}[/]개\n"
                f"압축 [dim]│[/] "
                f"{'[green]ON[/]' if self.resize_large_images else '[dim]OFF[/]'}"
                + (
                    f"  [dim]{self.resize_threshold_bytes / MB:.1f}MB 이상, "
                    f"긴 변 {self.resize_max_side}px, 품질 {self.resize_quality}[/]"
                    if self.resize_large_images else ""
                ),
                border_style="cyan",
                title="[bold cyan]설정[/]",
            )
        )
        console.print()
        self._emit_event("start", {
            "total_characters": total_chars,
            "min_images": min_images,
            "max_images": max_images,
            "workers": self.workers,
            "resize_large_images": self.resize_large_images,
            "resize_threshold_mb": round(self.resize_threshold_bytes / MB, 2),
            "resize_max_side": self.resize_max_side,
            "resize_quality": self.resize_quality,
            "total_target": total_target,
            "health": self._health_snapshot(),
            "updated_at": time.time(),
        })

        results: dict[str, list] = {
            "included":        [],
            "skipped":         [],
            "below_threshold": [],
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=26),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=self.emit_events,
        ) as progress:
            overall = progress.add_task(
                f"[bold cyan]{'전체 진행':<26}[/]",
                total=len(target),
            )

            for idx, (char_name, tag) in enumerate(target.items(), start=1):
                task_id = progress.add_task(
                    f"[yellow]{char_name:<26}[/]",
                    total=max_images,
                )

                status, count = self.crawl_character(
                    char_name, tag, max_images, min_images, progress, task_id, idx, total_chars
                )

                if status == "included":
                    progress.update(
                        task_id,
                        description=f"[green]✓ {char_name:<25}[/]",
                    )
                    results["included"].append((char_name, count))
                    self._run_context["included"] += 1

                elif status == "skipped":
                    progress.update(
                        task_id,
                        description=f"[dim]  {char_name:<25}[/]",
                    )
                    results["skipped"].append((char_name, count))
                    self._run_context["skipped"] += 1

                else:  # below_threshold
                    progress.update(
                        task_id,
                        description=f"[red]✗ {char_name:<25}[/]",
                    )
                    self._move_to_others(char_name)
                    results["below_threshold"].append((char_name, count))
                    self._run_context["below_threshold"] += 1

                progress.update(overall, advance=1)
                self._run_context["completed_characters"] += 1
                self._emit_progress(
                    char_name=char_name,
                    char_index=idx,
                    phase="done",
                    status=status,
                    collected=count,
                    target=max_images,
                    char_started_at=time.monotonic(),
                    force=True,
                )
                self._run_context["completed_images"] += min(count, max_images)

        self._print_summary(results)
        self._emit_event("complete", {
            "included": len(results["included"]),
            "skipped": len(results["skipped"]),
            "below_threshold": len(results["below_threshold"]),
            "total_downloaded": int(self._run_context.get("completed_images") or 0),
            "total_target": total_target,
            "health": self._health_snapshot(),
            "updated_at": time.time(),
        })
        return results

    def _print_summary(self, results: dict) -> None:
        console.print()
        console.rule("[bold]크롤링 결과[/]", style="cyan")
        console.print()

        if results["included"]:
            t = Table(
                title=f"포함된 캐릭터 ({len(results['included'])}명)",
                box=box.ROUNDED,
                border_style="green",
                header_style="bold green",
                show_lines=False,
            )
            t.add_column("캐릭터", style="white", min_width=26)
            t.add_column("이미지", justify="right", style="bold green")
            for name, cnt in sorted(results["included"], key=lambda x: -x[1]):
                t.add_row(name, f"{cnt:,}")
            console.print(t)

        if results["below_threshold"]:
            console.print()
            t2 = Table(
                title=f"기준 미달 → others/ ({len(results['below_threshold'])}명)",
                box=box.ROUNDED,
                border_style="yellow",
                header_style="bold yellow",
            )
            t2.add_column("캐릭터", style="dim", min_width=26)
            t2.add_column("이미지", justify="right", style="yellow")
            for name, cnt in results["below_threshold"]:
                t2.add_row(name, f"{cnt:,}")
            console.print(t2)

        if results["skipped"]:
            console.print(
                f"[dim]이미 완료된 캐릭터: {len(results['skipped'])}명 (스킵됨)[/]"
            )
        console.print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Danbooru 이미지 크롤러 (SFW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python danbooru_crawler.py                          # 익명 실행 (Hololive 전체)
  python danbooru_crawler.py -u USER -k KEY           # 인증 실행 (권장)
  python danbooru_crawler.py --min-images 300         # 최소 기준 완화
  python danbooru_crawler.py --tags-file tags.json    # 커스텀 캐릭터 크롤
                                                      # tags.json = {"char": "tag", ...}

자격증명은 .env 파일에 미리 설정해도 됩니다:
  DANBOORU_LOGIN=your_username
  DANBOORU_API_KEY=your_api_key
        """,
    )
    parser.add_argument(
        "-u", "--username",
        default=os.getenv("DANBOORU_LOGIN", ""),
        metavar="USER",
        help="Danbooru 사용자명 (또는 .env DANBOORU_LOGIN)",
    )
    parser.add_argument(
        "-k", "--api-key",
        default=os.getenv("DANBOORU_API_KEY", ""),
        metavar="KEY",
        help="Danbooru API 키 (또는 .env DANBOORU_API_KEY)",
    )
    parser.add_argument(
        "--min-images",
        type=int, default=500, metavar="N",
        help="캐릭터 포함 최소 이미지 수 (기본: 500)",
    )
    parser.add_argument(
        "--max-images",
        type=int, default=1000, metavar="N",
        help="캐릭터당 최대 수집 이미지 수 (기본: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset/raw", metavar="DIR",
        help="이미지 저장 경로 (기본: ./dataset/raw)",
    )
    parser.add_argument(
        "--workers",
        type=int, default=4, metavar="N",
        help="병렬 다운로드 스레드 수 (기본: 4, 최대 권장: 8)",
    )
    parser.add_argument(
        "--members",
        default="", metavar="KEYS",
        help="크롤할 캐릭터 키 (쉼표 구분, 기본: 전체). Hololive 키만 지원.",
    )
    parser.add_argument(
        "--tags-file",
        default="", metavar="FILE",
        help='커스텀 캐릭터 JSON 파일: {"key": "danbooru_tag", ...}. 지정 시 --members 무시.',
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Studio UI용 구조화 진행 이벤트를 출력하고 Rich 진행바를 비활성화합니다.",
    )
    parser.add_argument(
        "--resize-large-images",
        action="store_true",
        help="다운로드한 이미지가 기준 용량 이상이면 저장 전에 자동 축소/압축합니다.",
    )
    parser.add_argument(
        "--resize-threshold-mb",
        type=float,
        default=DEFAULT_RESIZE_THRESHOLD_MB,
        metavar="MB",
        help=f"자동 축소/압축 기준 용량 MB (기본: {DEFAULT_RESIZE_THRESHOLD_MB:g})",
    )
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=DEFAULT_RESIZE_MAX_SIDE,
        metavar="PX",
        help=f"자동 축소 시 긴 변 최대 픽셀 (기본: {DEFAULT_RESIZE_MAX_SIDE})",
    )
    parser.add_argument(
        "--resize-quality",
        type=int,
        default=DEFAULT_RESIZE_QUALITY,
        metavar="Q",
        help=f"JPEG/WebP 저장 품질 50-100 (기본: {DEFAULT_RESIZE_QUALITY})",
    )
    args = parser.parse_args()

    # 우선순위: --tags-file > --members > 전체(HOLOLIVE_MEMBERS)
    if args.tags_file.strip():
        import json as _json
        with open(args.tags_file, encoding="utf-8") as _f:
            members = _json.load(_f)   # {key: tag}
        console.print(f"[cyan]커스텀 태그 파일 로드:[/] {len(members)}개 캐릭터")
    elif args.members.strip():
        keys    = {k.strip() for k in args.members.split(",") if k.strip()}
        members = {k: v for k, v in HOLOLIVE_MEMBERS.items() if k in keys}
        unknown = keys - set(HOLOLIVE_MEMBERS)
        if unknown:
            console.print(f"[yellow]경고: 알 수 없는 캐릭터 키 무시됨: {unknown}[/]")
    else:
        members = None  # Hololive 전체 크롤

    DanbooruCrawler(
        username=args.username,
        api_key=args.api_key,
        output_dir=Path(args.output_dir),
        workers=args.workers,
        emit_events=args.events,
        resize_large_images=args.resize_large_images,
        resize_threshold_mb=args.resize_threshold_mb,
        resize_max_side=args.resize_max_side,
        resize_quality=args.resize_quality,
    ).run(min_images=args.min_images, max_images=args.max_images, members=members)
