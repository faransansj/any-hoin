/**
 * Dataset 페이지
 * - 좌측: 라벨 목록 (CRUD)
 * - 우측: 이미지 그리드 (선택 → 이동/삭제 / 업로드)
 */
import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { api } from "../api";
import { useTranslation } from "react-i18next";
import ImageModal from "../components/ImageModal";
import { useJobStore } from "../store/jobStore";
import type { DatasetDiscovery, ImageItem, ImageSort, JobState, Label } from "../types";

const PER_PAGE = 60;

export default function Dataset() {
  const { t } = useTranslation();
  const { crawlState, setCrawlState } = useJobStore();
  const [labels,       setLabels]       = useState<Label[]>([]);
  const [activeLabel,  setActiveLabel]  = useState<string | null>(null);
  const [images,       setImages]       = useState<ImageItem[]>([]);
  const [totalImages,  setTotalImages]  = useState(0);
  const [page,         setPage]         = useState(1);
  const [selected,     setSelected]     = useState<Set<string>>(new Set());
  const [previewImg,    setPreviewImg]    = useState<ImageItem | null>(null);
  const [newLabelName, setNewLabelName] = useState("");
  const [moveTarget,   setMoveTarget]   = useState("");
  const [loading,      setLoading]      = useState(false);
  const [discovery,    setDiscovery]    = useState<DatasetDiscovery | null>(null);
  const [recovering,   setRecovering]   = useState(false);
  const [labelSearch,  setLabelSearch]  = useState("");
  const [labelSort,    setLabelSort]    = useState<"name_asc" | "name_desc" | "count_desc" | "count_asc">("name_asc");
  const [lowOnly,      setLowOnly]      = useState(false);
  const [imageSort,    setImageSort]    = useState<ImageSort>("name_asc");
  const [lastSelectedIndex, setLastSelectedIndex] = useState<number | null>(null);

  // ── 라벨 목록 ──────────────────────────────────────────
  const loadLabels = useCallback(async () => {
    const r = await api.get<{ labels: Label[] }>("/labels");
    setLabels(r.labels);
    setActiveLabel((current) => {
      if (current && r.labels.some((l) => l.name === current)) return current;
      return r.labels[0]?.name ?? null;
    });
  }, []);

  const loadDatasetDiscovery = useCallback(async () => {
    const r = await api.get<DatasetDiscovery>("/characters/discover");
    setDiscovery(r);
  }, []);

  // ── 이미지 목록 ────────────────────────────────────────
  const loadImages = useCallback(async (label: string, p = 1) => {
    setLoading(true);
    try {
      const r = await api.get<{ total: number; images: ImageItem[] }>(
        `/images?label=${encodeURIComponent(label)}&page=${p}&per_page=${PER_PAGE}&sort=${imageSort}`
      );
      setImages(r.images);
      setTotalImages(r.total);
      setPage(p);
      setSelected(new Set());
      setLastSelectedIndex(null);
    } finally {
      setLoading(false);
    }
  }, [imageSort]);

  useEffect(() => {
    void loadLabels();
    void loadDatasetDiscovery();
  }, [loadDatasetDiscovery, loadLabels]);
  useEffect(() => {
    if (activeLabel) {
      void loadImages(activeLabel, 1);
      return;
    }
    setImages([]);
    setTotalImages(0);
    setPage(1);
    setSelected(new Set());
  }, [activeLabel, loadImages]);

  useEffect(() => {
    api.get<{ state: JobState }>("/crawl/status")
      .then((r) => setCrawlState(r.state))
      .catch(console.error);
  }, [setCrawlState]);

  useEffect(() => {
    if (crawlState !== "running") return;

    let dead = false;

    async function refreshDatasetFromCrawl() {
      try {
        const status = await api.get<{ state: JobState }>("/crawl/status");
        if (dead) return;
        setCrawlState(status.state);
        await loadLabels();
        await loadDatasetDiscovery();
        if (activeLabel) await loadImages(activeLabel, page);
      } catch (e) {
        console.error(e);
      }
    }

    void refreshDatasetFromCrawl();
    const timer = setInterval(refreshDatasetFromCrawl, 5000);

    return () => {
      dead = true;
      clearInterval(timer);
    };
  }, [activeLabel, crawlState, loadDatasetDiscovery, loadImages, loadLabels, page, setCrawlState]);

  // ── 라벨 생성 ─────────────────────────────────────────
  async function createLabel() {
    const name = newLabelName.trim();
    if (!name) return;
    await api.post("/labels", { name });
    setNewLabelName("");
    await loadLabels();
    await loadDatasetDiscovery();
    setActiveLabel(name);
  }

  async function deleteLabel(name: string) {
    if (!confirm(`"${name}" ${t("common.confirm_delete")}`)) return;
    await api.delete(`/labels/${encodeURIComponent(name)}`);
    if (activeLabel === name) setActiveLabel(null);
    await loadLabels();
    await loadDatasetDiscovery();
  }

  async function recoverDatasetCharacters() {
    if (!discovery?.missing.length) return;
    setRecovering(true);
    try {
      await api.post("/characters/recover", {
        keys: discovery.missing.map((item) => item.key),
      });
      await loadDatasetDiscovery();
    } finally {
      setRecovering(false);
    }
  }

  // ── 이미지 업로드 (드롭존) ─────────────────────────────
  const onDrop = useCallback(async (files: File[]) => {
    if (!activeLabel) return;
    const form = new FormData();
    form.append("label", activeLabel);
    files.forEach((f) => form.append("files", f));
    await api.upload("/images/upload", form);
    await loadLabels();
    await loadImages(activeLabel, page);
  }, [activeLabel, loadImages, loadLabels, page]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    noClick: true,
  });

  // ── 이미지 선택 ───────────────────────────────────────
  function toggleImage(id: string, index: number, range: boolean) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (range && lastSelectedIndex != null) {
        const [start, end] = [lastSelectedIndex, index].sort((a, b) => a - b);
        images.slice(start, end + 1).forEach((img) => next.add(img.id));
      } else {
        next.has(id) ? next.delete(id) : next.add(id);
      }
      return next;
    });
    setLastSelectedIndex(index);
  }

  function selectAll() {
    setSelected(new Set(images.map((i) => i.id)));
  }

  function clearSelection() {
    setSelected(new Set());
    setLastSelectedIndex(null);
  }

  function invertSelection() {
    setSelected((prev) => {
      const next = new Set<string>();
      images.forEach((img) => {
        if (!prev.has(img.id)) next.add(img.id);
      });
      return next;
    });
  }

  // ── 이동 / 삭제 ───────────────────────────────────────
  async function moveSelected() {
    if (!moveTarget || selected.size === 0) return;
    await api.post("/images/move", {
      image_ids:    Array.from(selected),
      target_label: moveTarget,
    });
    await loadLabels();
    await loadImages(activeLabel!, page);
  }

  async function deleteSelected() {
    if (selected.size === 0) return;
    if (!confirm(`${selected.size}장을 삭제하시겠습니까?`)) return;
    await api.delete("/images", { image_ids: Array.from(selected) });
    await loadLabels();
    await loadImages(activeLabel!, page);
  }

  const totalPages = Math.ceil(totalImages / PER_PAGE);
  const filteredLabels = labels
    .filter((label) => {
      const q = labelSearch.trim().toLowerCase();
      if (lowOnly && !label.warning) return false;
      return !q || label.name.toLowerCase().includes(q);
    })
    .sort((a, b) => {
      if (labelSort === "name_desc") return b.name.localeCompare(a.name);
      if (labelSort === "count_desc") return b.count - a.count;
      if (labelSort === "count_asc") return a.count - b.count;
      return a.name.localeCompare(b.name);
    });

  return (
    <>
    <div className="flex h-full" {...getRootProps()}>
      <input {...getInputProps()} />

      {/* ── 좌측 라벨 패널 ─────────────────────── */}
      <aside className="w-52 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col shrink-0">
	        <div className="p-3 border-b border-gray-200 dark:border-gray-800">
	          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">{t("dataset.labels_title")}</p>
	          <div className="flex gap-1">
            <input
              value={newLabelName}
              onChange={(e) => setNewLabelName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && createLabel()}
              placeholder={t("dataset.new_label_placeholder")}
              className="input text-xs py-1"
            />
	            <button onClick={createLabel} className="btn-primary px-2 py-1 text-xs shrink-0">+</button>
	          </div>
            <div className="mt-2 space-y-1.5">
              <input
                value={labelSearch}
                onChange={(e) => setLabelSearch(e.target.value)}
                className="input text-xs py-1"
                placeholder="라벨 검색..."
              />
              <select
                value={labelSort}
                onChange={(e) => setLabelSort(e.target.value as typeof labelSort)}
                className="input text-xs py-1"
              >
                <option value="name_asc">이름 오름차순</option>
                <option value="name_desc">이름 내림차순</option>
                <option value="count_desc">이미지 많은 순</option>
                <option value="count_asc">이미지 적은 순</option>
              </select>
              <label className="flex items-center gap-2 text-[11px] text-gray-500">
                <input
                  type="checkbox"
                  checked={lowOnly}
                  onChange={(e) => setLowOnly(e.target.checked)}
                  className="accent-brand-500"
                />
                부족 라벨만 보기
              </label>
            </div>
	        </div>

	        <div className="flex-1 overflow-y-auto py-2">
	          {filteredLabels.length === 0 && (
	            <p className="px-3 py-6 text-xs text-gray-500">조건에 맞는 라벨이 없습니다.</p>
	          )}
	          {filteredLabels.map((l) => (
	            <div
              key={l.name}
              onClick={() => setActiveLabel(l.name)}
              className={`group flex items-center px-3 py-2 cursor-pointer transition-colors ${
                activeLabel === l.name ? "bg-brand-600/20 text-white" : "hover:bg-gray-800 text-gray-400"
              }`}
            >
              <span className="flex-1 truncate text-sm">{l.name}</span>
              <span className={`text-xs mr-1 ${l.warning ? "text-yellow-400" : "text-gray-600"}`}>
                {l.count}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteLabel(l.name); }}
                className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 text-xs"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* ── 우측 이미지 그리드 ─────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900 shrink-0">
              <span className="text-sm text-gray-600 dark:text-gray-300 font-medium">
                {activeLabel ?? t("dataset.selected_label")}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-600">{totalImages}{t("dataset.total_images")}</span>
 
	              {selected.size > 0 && (
	                <>
	                  <span className="text-xs text-brand-400">{t("dataset.selected_count", { count: selected.size })}</span>
                  <select
                    value={moveTarget}
                    onChange={(e) => setMoveTarget(e.target.value)}
                    className="input text-xs py-1 w-32"
                  >
                    <option value="">{t("dataset.move_target")}</option>
                    {labels.filter((l) => l.name !== activeLabel).map((l) => (
                      <option key={l.name} value={l.name}>{l.name}</option>
                    ))}
                  </select>
                    <button onClick={moveSelected} disabled={!moveTarget} className="btn-ghost text-xs py-1">{t("common.move") || "이동"}</button>
                    <button onClick={deleteSelected} className="btn-danger text-xs py-1">{t("common.delete")}</button>
                    <button onClick={clearSelection} className="btn-ghost text-xs py-1">해제</button>
	                </>
	              )}

	              <div className="ml-auto flex items-center gap-2">
                  <select
                    value={imageSort}
                    onChange={(e) => setImageSort(e.target.value as ImageSort)}
                    className="input text-xs py-1 w-32"
                    disabled={!activeLabel}
                  >
                    <option value="name_asc">이름순</option>
                    <option value="name_desc">이름 역순</option>
                    <option value="newest">최신순</option>
                    <option value="oldest">오래된순</option>
                  </select>
	                <button onClick={selectAll} className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">{t("dataset.select_all")}</button>
                  <button onClick={invertSelection} className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">반전</button>
	                <label className="btn-ghost text-xs py-1 cursor-pointer">
                  {t("dataset.upload_btn")}
                  <input type="file" multiple accept="image/*" className="hidden"
                    onChange={(e) => e.target.files && onDrop(Array.from(e.target.files))} />
                </label>
              </div>
            </div>

            {discovery && discovery.missing.length > 0 && (
              <div className="mx-4 mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 flex items-start justify-between gap-3 shrink-0">
                <div>
                  <p className="text-xs font-semibold text-amber-300">
                    기존 이미지 폴더 {discovery.missing.length}개가 characters.json에 등록되어 있지 않습니다.
                  </p>
                  <p className="text-[11px] text-amber-100/70 mt-1">
                    세션 종료로 이미지가 삭제된 것은 아닙니다. 필요하면 현재 dataset/raw 폴더를 다시 캐릭터 목록에 등록하세요.
                  </p>
                </div>
                <button
                  onClick={recoverDatasetCharacters}
                  disabled={recovering}
                  className="shrink-0 text-xs px-3 py-1.5 rounded bg-amber-500 text-gray-950 hover:bg-amber-400 disabled:opacity-50"
                >
                  {recovering ? "불러오는 중" : "데이터셋 불러오기"}
                </button>
              </div>
            )}


        {/* 그리드 */}
        <div className="flex-1 overflow-y-auto p-4">
           {isDragActive && (
             <div className="absolute inset-0 bg-brand-600/10 border-2 border-brand-500 border-dashed rounded-xl flex items-center justify-center z-10">
               <p className="text-brand-400 text-lg font-medium">{t("dataset.drop_hint")}</p>
             </div>
           )}
 
           {loading ? (
             <div className="flex items-center justify-center h-32 text-gray-500 dark:text-gray-600 text-sm">{t("common.loading")}</div>
           ) : images.length === 0 ? (
             <div className="flex flex-col items-center justify-center h-32 text-gray-500 dark:text-gray-600 text-sm gap-2">
               <span>{t("dataset.empty_images")}</span>
               <span className="text-xs">{t("dataset.empty_hint")}</span>
             </div>
           ) : (

            <div className="grid grid-cols-6 gap-2">
              {images.map((img, index) => {
                const sel = selected.has(img.id);
                return (
                     <div
                       key={img.id}
	                       onClick={(e) => toggleImage(img.id, index, e.shiftKey)}
                       className={`group relative aspect-square rounded-lg overflow-hidden cursor-pointer border-2 transition-all ${
                         sel ? "border-brand-500 scale-95" : "border-transparent hover:border-gray-600"
                       }`}
                     >
                    <img
                      src={img.thumbnail}
                      alt={img.name}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                    {sel && (
                      <div className="absolute inset-0 bg-brand-600/30 flex items-center justify-center">
                        <span className="text-white text-lg">✓</span>
                      </div>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); setPreviewImg(img); }}
                      className="absolute top-1 right-1 p-1 rounded bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80"
                      title="전체보기"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-5h-4m4 0v4m0-4l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                      </svg>
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* 페이지네이션 */}
         {totalPages > 1 && (
           <div className="flex items-center justify-center gap-2 py-3 border-t border-gray-200 dark:border-gray-800 shrink-0">
             <button onClick={() => loadImages(activeLabel!, page - 1)} disabled={page <= 1}
               className="btn-ghost text-xs py-1 px-3">{t("dataset.prev")}</button>
             <span className="text-xs text-gray-500 dark:text-gray-400">{page} / {totalPages}</span>
             <button onClick={() => loadImages(activeLabel!, page + 1)} disabled={page >= totalPages}
               className="btn-ghost text-xs py-1 px-3">{t("dataset.next")}</button>
           </div>
         )}

       </div>
     </div>
     {previewImg && (
       <ImageModal
         src={previewImg.thumbnail}
         alt={previewImg.name}
         onClose={() => setPreviewImg(null)}
       />
     )}
    </>
   );
 }
