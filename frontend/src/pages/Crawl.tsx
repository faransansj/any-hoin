/**
 * Crawl 페이지
 * - 캐릭터 관리 (추가/수정/삭제)
 * - 크롤 대상 선택 + 파라미터 설정
 * - 크롤 실행/중단 + 실시간 로그
 */
import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import JobConsole from "../components/JobConsole";
import StatusBadge from "../components/StatusBadge";
import TagSearchInput from "../components/TagSearchInput";
import { useJobStore } from "../store/jobStore";
import type { Character, JobState } from "../types";

interface EditState {
  key: string;
  tag: string;
  display_name: string;
  postCount?: number | null;
}

export default function Crawl() {
  const { crawlState, setCrawlState } = useJobStore();

  const [characters, setCharacters] = useState<Character[]>([]);
  const [selected,   setSelected]   = useState<Set<string>>(new Set());

  // 새 캐릭터 추가 폼
  const [addOpen,       setAddOpen]       = useState(false);
  const [newKey,        setNewKey]        = useState("");
  const [newTag,        setNewTag]        = useState("");
  const [newName,       setNewName]       = useState("");
  const [newPostCount,  setNewPostCount]  = useState<number | null>(null);
  const [addErr,        setAddErr]        = useState("");

  // 인라인 편집
  const [editing, setEditing]   = useState<EditState | null>(null);

  // 크롤 설정
  const [minImages, setMinImages] = useState(500);
  const [maxImages, setMaxImages] = useState(1000);
  const [workers,   setWorkers]   = useState(4);
  const [username,  setUsername]  = useState("");
  const [apiKey,    setApiKey]    = useState("");

  const running = crawlState === "running";

  // 목록 로드
  function loadCharacters() {
    api.get<{ characters: Character[] }>("/characters")
      .then((r) => setCharacters(r.characters))
      .catch(console.error);
  }

  useEffect(() => {
    loadCharacters();
    api.get<{ state: JobState }>("/crawl/status")
      .then((r) => setCrawlState(r.state))
      .catch(console.error);
  }, []);

  // ── 선택 ────────────────────────────────────────────────
  const allSelected = characters.length > 0 && characters.every((c) => selected.has(c.key));

  function toggleAll() {
    if (allSelected) {
      setSelected(new Set());
    } else {
      setSelected(new Set(characters.map((c) => c.key)));
    }
  }

  function toggle(key: string) {
    const next = new Set(selected);
    next.has(key) ? next.delete(key) : next.add(key);
    setSelected(next);
  }

  // 태그 드롭다운 선택 시 빈 key/이름 자동 채우기
  function handleTagSelect(tag: string) {
    if (!newKey.trim())  setNewKey(tag);
    if (!newName.trim()) setNewName(tag);
  }

  // ── 캐릭터 추가 ─────────────────────────────────────────
  async function handleAdd() {
    setAddErr("");
    const tag  = newTag.trim();
    const key  = newKey.trim().replace(/\s+/g, "_").toLowerCase() || tag;
    const name = newName.trim() || tag;
    if (!key) { setAddErr("Key를 입력하세요"); return; }
    if (!tag) { setAddErr("Danbooru 태그를 입력하세요"); return; }
    try {
      await api.post("/characters", { key, tag, display_name: name });
      setNewKey(""); setNewTag(""); setNewName(""); setNewPostCount(null); setAddOpen(false);
      loadCharacters();
    } catch (e: any) {
      setAddErr(e?.message ?? "추가 실패");
    }
  }

  // ── 캐릭터 삭제 ─────────────────────────────────────────
  async function handleDelete(key: string) {
    if (!confirm(`'${key}'를 삭제할까요?`)) return;
    await api.delete(`/characters/${key}`);
    setSelected((prev) => { const s = new Set(prev); s.delete(key); return s; });
    loadCharacters();
  }

  // ── 인라인 편집 저장 ─────────────────────────────────────
  async function handleEditSave() {
    if (!editing) return;
    await api.put(`/characters/${editing.key}`, {
      tag:          editing.tag,
      display_name: editing.display_name,
    });
    setEditing(null);
    loadCharacters();
  }

  // ── 크롤 시작 ────────────────────────────────────────────
  async function startCrawl() {
    const selectedKeys = selected.size > 0 ? Array.from(selected) : characters.map((c) => c.key);
    await api.post("/crawl/start", {
      selected_keys: selectedKeys,
      min_images:    minImages,
      max_images:    maxImages,
      workers,
      username:      username || undefined,
      api_key:       apiKey   || undefined,
    });
  }

  async function stopCrawl() {
    await api.post("/crawl/stop");
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* ── 헤더 ── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Crawl</h1>
          <p className="text-sm text-gray-400 mt-0.5">Danbooru에서 학습 이미지 수집</p>
        </div>
        <StatusBadge state={crawlState} />
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* ── 캐릭터 관리 ── */}
        <div className="col-span-2 card space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-200">캐릭터</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500">
                {selected.size > 0 ? `${selected.size}개 선택 /` : ""} 전체 {characters.length}개
              </span>
              <button
                onClick={toggleAll}
                className="text-xs text-brand-400 hover:text-brand-300"
              >
                {allSelected ? "전체 해제" : "전체 선택"}
              </button>
              <button
                onClick={() => { setAddOpen((o) => !o); setAddErr(""); }}
                className="text-xs px-2 py-1 rounded bg-brand-600 hover:bg-brand-500 text-white transition-colors"
              >
                + 추가
              </button>
            </div>
          </div>

          {/* 추가 폼 */}
          {addOpen && (
            <div className="bg-gray-800 rounded-lg p-3 space-y-2 border border-gray-700">
              <p className="text-xs font-medium text-gray-300">새 캐릭터 추가</p>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="label-text">Key (폴더명)</label>
                  <input
                    value={newKey}
                    onChange={(e) => setNewKey(e.target.value)}
                    className="input text-xs"
                    placeholder="tokino_sora"
                  />
                  <p className="text-[10px] text-gray-600 mt-0.5">영문·숫자·_ 권장</p>
                </div>
                <div>
                  <label className="label-text">표시 이름</label>
                  <input
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    className="input text-xs"
                    placeholder="Tokino Sora (선택)"
                  />
                </div>
                <div>
                  <label className="label-text">Danbooru 태그</label>
                  <TagSearchInput
                    value={newTag}
                    onChange={setNewTag}
                    onPostCount={setNewPostCount}
                    onSelect={handleTagSelect}
                    placeholder="태그 입력 또는 검색..."
                  />
                  {newPostCount != null && (
                    <p className="text-[10px] text-green-500 mt-0.5">
                      Danbooru {newPostCount.toLocaleString()}장 확인됨
                    </p>
                  )}
                </div>
              </div>
              {addErr && <p className="text-xs text-red-400">{addErr}</p>}
              <div className="flex gap-2">
                <button onClick={handleAdd} className="btn-primary text-xs px-3 py-1">추가</button>
                <button
                  onClick={() => { setAddOpen(false); setAddErr(""); setNewPostCount(null); }}
                  className="text-xs px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
                >
                  취소
                </button>
              </div>
            </div>
          )}

          {/* 캐릭터 테이블 */}
          <div className="max-h-72 overflow-y-auto rounded-lg border border-gray-700">
            {characters.length === 0 ? (
              <p className="text-xs text-gray-500 text-center py-8">
                캐릭터가 없습니다. 추가 버튼으로 캐릭터를 등록하세요.
              </p>
            ) : (
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-800 text-gray-400">
                  <tr>
                    <th className="w-8 px-2 py-2"></th>
                    <th className="text-left px-2 py-2">Key</th>
                    <th className="text-left px-2 py-2">표시 이름</th>
                    <th className="text-left px-2 py-2">Danbooru 태그</th>
                    <th className="text-right px-2 py-2">이미지</th>
                    <th className="w-16 px-2 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {characters.map((c) => {
                    const sel = selected.has(c.key);
                    const isEditing = editing?.key === c.key;
                    return (
                      <tr
                        key={c.key}
                        className={`border-t border-gray-700/50 transition-colors ${
                          sel ? "bg-brand-600/10" : "hover:bg-gray-800/50"
                        }`}
                      >
                        <td className="px-2 py-1.5 text-center">
                          <input
                            type="checkbox"
                            checked={sel}
                            onChange={() => toggle(c.key)}
                            className="accent-brand-500"
                          />
                        </td>
                        <td className="px-2 py-1.5 text-gray-300 font-mono">{c.key}</td>
                        {isEditing ? (
                          <>
                            <td className="px-2 py-1">
                              <input
                                value={editing.display_name}
                                onChange={(e) => setEditing({ ...editing, display_name: e.target.value })}
                                className="input text-xs py-0.5"
                              />
                            </td>
                            <td className="px-2 py-1 min-w-[160px]">
                              <TagSearchInput
                                value={editing.tag}
                                onChange={(v) => setEditing({ ...editing, tag: v })}
                                onPostCount={(n) => setEditing({ ...editing, postCount: n })}
                                className="input text-xs py-0.5 w-full"
                              />
                              {editing.postCount != null && (
                                <p className="text-[10px] text-green-500 mt-0.5">
                                  {editing.postCount.toLocaleString()}장
                                </p>
                              )}
                            </td>
                          </>
                        ) : (
                          <>
                            <td className="px-2 py-1.5 text-gray-200">{c.display_name}</td>
                            <td className="px-2 py-1.5 text-gray-400 font-mono">{c.tag}</td>
                          </>
                        )}
                        <td className="px-2 py-1.5 text-right text-gray-500">{c.count}</td>
                        <td className="px-2 py-1.5 text-right">
                          {isEditing ? (
                            <div className="flex gap-1 justify-end">
                              <button
                                onClick={handleEditSave}
                                className="text-brand-400 hover:text-brand-300"
                              >
                                저장
                              </button>
                              <button
                                onClick={() => setEditing(null)}
                                className="text-gray-500 hover:text-gray-300"
                              >
                                취소
                              </button>
                            </div>
                          ) : (
                            <div className="flex gap-1 justify-end">
                              <button
                                onClick={() => setEditing({ key: c.key, tag: c.tag, display_name: c.display_name })}
                                className="text-gray-500 hover:text-gray-300"
                                title="수정"
                              >
                                ✎
                              </button>
                              <button
                                onClick={() => handleDelete(c.key)}
                                className="text-gray-600 hover:text-red-400"
                                title="삭제"
                              >
                                ✕
                              </button>
                            </div>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* ── 설정 ── */}
        <div className="card space-y-3">
          <p className="text-sm font-medium text-gray-200">설정</p>

          <div>
            <label className="label-text">최소 이미지 수</label>
            <input
              type="number" value={minImages}
              onChange={(e) => setMinImages(+e.target.value)}
              className="input" min={1} disabled={running}
            />
          </div>
          <div>
            <label className="label-text">최대 이미지 수</label>
            <input
              type="number" value={maxImages}
              onChange={(e) => setMaxImages(+e.target.value)}
              className="input" min={1} disabled={running}
            />
          </div>
          <div>
            <label className="label-text">병렬 워커 수</label>
            <input
              type="number" value={workers}
              onChange={(e) => setWorkers(+e.target.value)}
              className="input" min={1} max={16} disabled={running}
            />
          </div>

          <hr className="border-gray-700" />

          <div>
            <label className="label-text">Danbooru ID (선택)</label>
            <input
              value={username} onChange={(e) => setUsername(e.target.value)}
              className="input" placeholder=".env 로드됨" disabled={running}
            />
          </div>
          <div>
            <label className="label-text">API Key (선택)</label>
            <input
              type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
              className="input" placeholder=".env 로드됨" disabled={running}
            />
          </div>

          <p className="text-xs text-gray-500">
            {selected.size === 0
              ? "선택 없으면 전체 크롤"
              : `${selected.size}개 캐릭터 크롤`}
          </p>

          <div className="pt-1 space-y-2">
            {!running ? (
              <button
                onClick={startCrawl}
                disabled={characters.length === 0}
                className="btn-primary w-full disabled:opacity-40"
              >
                크롤 시작
              </button>
            ) : (
              <button onClick={stopCrawl} className="btn-danger w-full">
                중단
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ── 로그 ── */}
      <div className="card">
        <p className="text-sm font-medium text-gray-200 mb-2">로그</p>
        <JobConsole
          jobPath="/crawl/logs"
          onState={(s) => setCrawlState(s)}
        />
      </div>
    </div>
  );
}
