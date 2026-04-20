/**
 * Dataset 페이지
 * - 좌측: 라벨 목록 (CRUD)
 * - 우측: 이미지 그리드 (선택 → 이동/삭제 / 업로드)
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";
import { api } from "../api";
import ImageModal from "../components/ImageModal";
import type { ImageItem, Label } from "../types";

export default function Dataset() {
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

  const PER_PAGE = 60;

  // ── 라벨 목록 ──────────────────────────────────────────
  async function loadLabels() {
    const r = await api.get<{ labels: Label[] }>("/labels");
    setLabels(r.labels);
    if (!activeLabel && r.labels.length > 0) {
      setActiveLabel(r.labels[0].name);
    }
  }

  // ── 이미지 목록 ────────────────────────────────────────
  async function loadImages(label: string, p = 1) {
    setLoading(true);
    try {
      const r = await api.get<{ total: number; images: ImageItem[] }>(
        `/images?label=${encodeURIComponent(label)}&page=${p}&per_page=${PER_PAGE}`
      );
      setImages(r.images);
      setTotalImages(r.total);
      setPage(p);
      setSelected(new Set());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadLabels(); }, []);
  useEffect(() => {
    if (activeLabel) loadImages(activeLabel, 1);
  }, [activeLabel]);

  // ── 라벨 생성 ─────────────────────────────────────────
  async function createLabel() {
    const name = newLabelName.trim();
    if (!name) return;
    await api.post("/labels", { name });
    setNewLabelName("");
    await loadLabels();
    setActiveLabel(name);
  }

  async function deleteLabel(name: string) {
    if (!confirm(`"${name}" 라벨과 이미지를 모두 삭제하시겠습니까?`)) return;
    await api.delete(`/labels/${encodeURIComponent(name)}`);
    if (activeLabel === name) setActiveLabel(null);
    await loadLabels();
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
  }, [activeLabel, page]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    noClick: true,
  });

  // ── 이미지 선택 ───────────────────────────────────────
  function toggleImage(id: string) {
    const next = new Set(selected);
    next.has(id) ? next.delete(id) : next.add(id);
    setSelected(next);
  }

  function selectAll() {
    setSelected(new Set(images.map((i) => i.id)));
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

  return (
    <>
    <div className="flex h-full" {...getRootProps()}>
      <input {...getInputProps()} />

      {/* ── 좌측 라벨 패널 ─────────────────────── */}
      <aside className="w-52 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
        <div className="p-3 border-b border-gray-800">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Labels</p>
          <div className="flex gap-1">
            <input
              value={newLabelName}
              onChange={(e) => setNewLabelName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && createLabel()}
              placeholder="새 라벨..."
              className="input text-xs py-1"
            />
            <button onClick={createLabel} className="btn-primary px-2 py-1 text-xs shrink-0">+</button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-2">
          {labels.map((l) => (
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
        {/* 툴바 */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-gray-800 bg-gray-900 shrink-0">
          <span className="text-sm text-gray-300 font-medium">
            {activeLabel ?? "라벨을 선택하세요"}
          </span>
          <span className="text-xs text-gray-600">{totalImages}장</span>

          {selected.size > 0 && (
            <>
              <span className="text-xs text-brand-400">{selected.size}개 선택됨</span>
              <select
                value={moveTarget}
                onChange={(e) => setMoveTarget(e.target.value)}
                className="input text-xs py-1 w-32"
              >
                <option value="">이동 대상...</option>
                {labels.filter((l) => l.name !== activeLabel).map((l) => (
                  <option key={l.name} value={l.name}>{l.name}</option>
                ))}
              </select>
              <button onClick={moveSelected} disabled={!moveTarget} className="btn-ghost text-xs py-1">이동</button>
              <button onClick={deleteSelected} className="btn-danger text-xs py-1">삭제</button>
            </>
          )}

          <div className="ml-auto flex items-center gap-2">
            <button onClick={selectAll} className="text-xs text-gray-400 hover:text-gray-200">전체 선택</button>
            <label className="btn-ghost text-xs py-1 cursor-pointer">
              업로드
              <input type="file" multiple accept="image/*" className="hidden"
                onChange={(e) => e.target.files && onDrop(Array.from(e.target.files))} />
            </label>
          </div>
        </div>

        {/* 그리드 */}
        <div className="flex-1 overflow-y-auto p-4">
          {isDragActive && (
            <div className="absolute inset-0 bg-brand-600/10 border-2 border-brand-500 border-dashed rounded-xl flex items-center justify-center z-10">
              <p className="text-brand-400 text-lg font-medium">이미지를 놓으세요</p>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center h-32 text-gray-600 text-sm">로딩 중...</div>
          ) : images.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-gray-600 text-sm gap-2">
              <span>이미지 없음</span>
              <span className="text-xs">이미지를 드래그하거나 업로드 버튼을 사용하세요</span>
            </div>
          ) : (
            <div className="grid grid-cols-6 gap-2">
              {images.map((img) => {
                const sel = selected.has(img.id);
                return (
                     <div
                       key={img.id}
                       onClick={() => toggleImage(img.id)}
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
          <div className="flex items-center justify-center gap-2 py-3 border-t border-gray-800 shrink-0">
            <button onClick={() => loadImages(activeLabel!, page - 1)} disabled={page <= 1}
              className="btn-ghost text-xs py-1 px-3">이전</button>
            <span className="text-xs text-gray-400">{page} / {totalPages}</span>
            <button onClick={() => loadImages(activeLabel!, page + 1)} disabled={page >= totalPages}
              className="btn-ghost text-xs py-1 px-3">다음</button>
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
