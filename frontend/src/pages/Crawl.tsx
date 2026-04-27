/**
 * Crawl 페이지
 * - 캐릭터 관리 (추가/수정/삭제)
 * - 크롤 대상 선택 + 파라미터 설정
 * - 크롤 실행/중단 + 실시간 로그
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import JobConsole from "../components/JobConsole";
import StatusBadge from "../components/StatusBadge";
import TagSearchInput from "../components/TagSearchInput";
import { useJobStore } from "../store/jobStore";
import { useTranslation } from "react-i18next";
import type {
  Character,
  CrawlHealthResponse,
  CrawlProgress,
  CrawlStatus,
  DatasetDiscovery,
  GenreCharacterCandidate,
  GenreCharactersResponse,
  JobState,
} from "../types";

interface EditState {
  key: string;
  tag: string;
  display_name: string;
  postCount?: number | null;
}

function fmtEta(sec: number | null | undefined): string {
  if (sec == null || sec < 0) return "계산 중";
  if (sec === 0) return "0s";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtPhase(phase: string): string {
  if (phase === "starting") return "시작";
  if (phase === "collecting") return "목록 수집";
  if (phase === "downloading") return "다운로드";
  if (phase === "done") return "완료";
  return phase;
}

function ProgressBar({ pct, color = "bg-brand-500" }: { pct: number; color?: string }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-gray-800">
      <div
        className={`${color} h-full rounded-full transition-all duration-300`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  );
}

export default function Crawl() {
  const { t } = useTranslation();
  const { crawlState, setCrawlState } = useJobStore();

  const [characters, setCharacters] = useState<Character[]>([]);
  const [selected,   setSelected]   = useState<Set<string>>(new Set());
  const [discovery,  setDiscovery]  = useState<DatasetDiscovery | null>(null);
  const [recovering, setRecovering] = useState(false);
  const [charSearch, setCharSearch] = useState("");
  const [charFilter, setCharFilter] = useState<"all" | "selected" | "under_min" | "others">("all");
  const [crawlProgress, setCrawlProgress] = useState<CrawlProgress | null>(null);
  const [crawlHealth, setCrawlHealth] = useState<CrawlHealthResponse | null>(null);
  const [healthChecking, setHealthChecking] = useState(false);
  const [healthStreaming, setHealthStreaming] = useState(true);

  // 새 캐릭터 추가 폼
  const [addOpen,       setAddOpen]       = useState(false);
  const [newKey,        setNewKey]        = useState("");
  const [newTag,        setNewTag]        = useState("");
  const [newName,       setNewName]       = useState("");
  const [newPostCount,  setNewPostCount]  = useState<number | null>(null);
  const [addErr,        setAddErr]        = useState("");
  const [genreQuery,      setGenreQuery]      = useState("");
  const [genreCandidates, setGenreCandidates] = useState<GenreCharacterCandidate[]>([]);
  const [genreSelected,   setGenreSelected]   = useState<Set<string>>(new Set());
  const [genreLoading,    setGenreLoading]    = useState(false);
  const [genreErr,        setGenreErr]        = useState("");
  const [genreNotice,     setGenreNotice]     = useState("");

  // 인라인 편집
  const [editing, setEditing]   = useState<EditState | null>(null);

  // 크롤 설정
  const [minImages, setMinImages] = useState(500);
  const [maxImages, setMaxImages] = useState(1000);
  const [workers,   setWorkers]   = useState(4);
  const [username,  setUsername]  = useState("");
  const [apiKey,    setApiKey]    = useState("");

  const running = crawlState === "running";
  const filteredCharacters = characters.filter((c) => {
    const q = charSearch.trim().toLowerCase();
    const matchesQuery = !q || [c.key, c.tag, c.display_name].some((v) => v.toLowerCase().includes(q));
    if (!matchesQuery) return false;
    if (charFilter === "selected") return selected.has(c.key);
    if (charFilter === "under_min") return c.count < minImages;
    if (charFilter === "others") return (c.other_count ?? 0) > 0;
    return true;
  });
  const visibleAllSelected = filteredCharacters.length > 0 && filteredCharacters.every((c) => selected.has(c.key));

  // 목록 로드
  function loadCharacters() {
    api.get<{ characters: Character[] }>("/characters")
      .then((r) => setCharacters(r.characters))
      .catch(console.error);
  }

  function loadDatasetDiscovery() {
    api.get<DatasetDiscovery>("/characters/discover")
      .then(setDiscovery)
      .catch(console.error);
  }

  useEffect(() => {
    loadCharacters();
    loadDatasetDiscovery();
    api.get<CrawlStatus>("/crawl/status")
      .then((r) => {
        setCrawlState(r.state);
        setCrawlProgress(r.current_progress);
        setCrawlHealth({
          state: r.state,
          heartbeat_ok: r.state !== "running" || (r.last_event_age_sec != null && r.last_event_age_sec < 45),
          last_event_age_sec: r.last_event_age_sec,
          crawler: r.health,
          current_progress: r.current_progress,
        });
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!healthStreaming) return;
    let dead = false;

    async function syncCrawlerHealth() {
      setHealthChecking(true);
      try {
        const r = await api.get<CrawlHealthResponse>("/crawl/health");
        if (dead) return;
        setCrawlHealth(r);
        if (r.current_progress) setCrawlProgress(r.current_progress);
      } catch (e) {
        if (!dead) console.error(e);
      } finally {
        if (!dead) setHealthChecking(false);
      }
    }

    void syncCrawlerHealth();
    const timer = setInterval(syncCrawlerHealth, 10000);
    return () => {
      dead = true;
      clearInterval(timer);
    };
  }, [healthStreaming]);

  // ── 선택 ────────────────────────────────────────────────
  function toggleVisible() {
    const keys = filteredCharacters.map((c) => c.key);
    setSelected((prev) => {
      const next = new Set(prev);
      if (visibleAllSelected) {
        keys.forEach((key) => next.delete(key));
      } else {
        keys.forEach((key) => next.add(key));
      }
      return next;
    });
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

  async function searchGenreCharacters() {
    const q = genreQuery.trim();
    if (q.length < 2 || genreLoading) return;
    setGenreLoading(true);
    setGenreErr("");
    setGenreNotice("");
    try {
      const r = await api.get<GenreCharactersResponse>(
        `/crawl/tags/genre?q=${encodeURIComponent(q)}&limit=120`
      );
      setGenreCandidates(r.characters);
      const existing = new Set(characters.map((c) => c.key));
      setGenreSelected(new Set(r.characters.filter((c) => !existing.has(c.key)).map((c) => c.key)));
      if (r.characters.length === 0) {
        setGenreErr("해당 장르에서 캐릭터 태그를 찾지 못했습니다.");
      }
    } catch (e: any) {
      setGenreCandidates([]);
      setGenreSelected(new Set());
      setGenreErr(e?.message ?? "장르 검색 실패");
    } finally {
      setGenreLoading(false);
    }
  }

  function toggleGenreCandidate(key: string) {
    setGenreSelected((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }

  function toggleAllGenreCandidates() {
    const importable = genreCandidates
      .filter((candidate) => !characters.some((c) => c.key === candidate.key))
      .map((candidate) => candidate.key);
    const allChecked = importable.length > 0 && importable.every((key) => genreSelected.has(key));
    setGenreSelected(allChecked ? new Set() : new Set(importable));
  }

  async function importGenreCharacters() {
    const items = genreCandidates.filter((candidate) => genreSelected.has(candidate.key));
    if (items.length === 0) return;
    const r = await api.post<{ imported: number; skipped: number; keys: string[] }>("/characters/import", {
      overwrite: false,
      characters: items.map((item) => ({
        key: item.key,
        tag: item.tag,
        display_name: item.display_name,
      })),
    });
    setGenreNotice(`${r.imported}개 추가됨, ${r.skipped}개 건너뜀`);
    setGenreSelected(new Set());
    loadCharacters();
    loadDatasetDiscovery();
  }

  // ── 캐릭터 추가 ─────────────────────────────────────────
  async function handleAdd() {
    setAddErr("");
    const tag  = newTag.trim();
    const key  = newKey.trim().replace(/\s+/g, "_").toLowerCase() || tag;
    const name = newName.trim() || tag;
    if (!key) { setAddErr(t("crawl.err_key")); return; }
    if (!tag) { setAddErr(t("crawl.err_tag")); return; }
    try {
      await api.post("/characters", { key, tag, display_name: name });
      setNewKey(""); setNewTag(""); setNewName(""); setNewPostCount(null); setAddOpen(false);
      loadCharacters();
      loadDatasetDiscovery();
    } catch (e: any) {
      setAddErr(e?.message ?? t("crawl.err_fail"));
    }
  }

  // ── 캐릭터 삭제 ─────────────────────────────────────────
  async function handleDelete(key: string) {
    if (!confirm(`'${key}'${t("common.confirm_delete")}`)) return;
    await api.delete(`/characters/${key}`);
    setSelected((prev) => { const s = new Set(prev); s.delete(key); return s; });
    loadCharacters();
    loadDatasetDiscovery();
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
    loadDatasetDiscovery();
  }

  async function recoverDatasetCharacters() {
    if (!discovery?.missing.length) return;
    setRecovering(true);
    try {
      await api.post("/characters/recover", {
        keys: discovery.missing.map((item) => item.key),
      });
      loadCharacters();
      loadDatasetDiscovery();
    } finally {
      setRecovering(false);
    }
  }

  // ── 크롤 시작 ────────────────────────────────────────────
  async function startCrawl() {
    const selectedKeys = selected.size > 0 ? Array.from(selected) : characters.map((c) => c.key);
    const result = await api.post<{ started?: boolean; error?: string }>("/crawl/start", {
      selected_keys: selectedKeys,
      min_images:    minImages,
      max_images:    maxImages,
      workers,
      username:      username || undefined,
      api_key:       apiKey   || undefined,
    });
    if (result.error) throw new Error(result.error);
    if (result.started) setCrawlState("running");
  }

  async function stopCrawl() {
    await api.post("/crawl/stop");
  }

  function handleCrawlState(state: JobState) {
    setCrawlState(state);
    if (state === "done" || state === "failed") {
      loadCharacters();
      loadDatasetDiscovery();
    }
  }

  function handleCrawlProgress(progress: CrawlProgress) {
    setCrawlProgress(progress);
    setCrawlHealth((prev) => ({
      state: "running",
      heartbeat_ok: true,
      last_event_age_sec: 0,
      crawler: progress.health,
      current_progress: progress,
      remote: prev?.remote,
    }));
  }

  const crawlerProgressPanel = (
    <div className="card space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold text-gray-200">크롤러 진행 상황</p>
          <p className="text-xs text-gray-500 mt-0.5">
            현재 캐릭터, 캐릭터별/전체 이미지 수, ETA, Danbooru rate limit/health 상태를 표시합니다.
          </p>
        </div>
        <button
          onClick={() => setHealthStreaming((enabled) => !enabled)}
          className={`text-xs px-3 py-1.5 rounded-lg shrink-0 transition-colors ${
            healthStreaming
              ? "bg-green-500/15 text-green-300 hover:bg-green-500/25"
              : "bg-gray-800 text-gray-400 hover:bg-gray-700"
          }`}
        >
          {healthStreaming ? (healthChecking ? "Health 실시간 확인 중" : "Health 실시간 켜짐") : "Health 실시간 꺼짐"}
        </button>
      </div>

      {crawlProgress ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">현재 캐릭터</p>
              <p className="mt-1 text-sm font-semibold text-white truncate">{crawlProgress.current_character}</p>
              <p className="mt-1 text-[11px] text-gray-500">
                {crawlProgress.current_index} / {crawlProgress.total_characters} · {fmtPhase(crawlProgress.phase)}
              </p>
            </div>
            <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">캐릭터 이미지</p>
              <p className="mt-1 text-sm font-semibold text-white tabular-nums">
                {crawlProgress.char_downloaded.toLocaleString()} / {crawlProgress.char_target.toLocaleString()}
              </p>
              <p className="mt-1 text-[11px] text-gray-500">남은 시간 {fmtEta(crawlProgress.char_eta_sec)}</p>
            </div>
            <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">전체 이미지</p>
              <p className="mt-1 text-sm font-semibold text-white tabular-nums">
                {crawlProgress.total_downloaded.toLocaleString()} / {crawlProgress.total_target.toLocaleString()}
              </p>
              <p className="mt-1 text-[11px] text-gray-500">남은 시간 {fmtEta(crawlProgress.overall_eta_sec)}</p>
            </div>
            <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">Danbooru</p>
              <p className={`mt-1 text-sm font-semibold ${crawlProgress.health.rate_limited ? "text-red-300" : "text-green-300"}`}>
                {crawlProgress.health.rate_limited ? "Rate limited" : "OK"}
              </p>
              <p className="mt-1 text-[11px] text-gray-500">
                HTTP {crawlProgress.health.last_status ?? "—"} · {crawlProgress.health.last_latency_ms ?? "—"}ms
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-xs text-gray-400">
              <span>현재 캐릭터 진행률</span>
              <span className="tabular-nums">{crawlProgress.char_pct.toFixed(1)}%</span>
            </div>
            <ProgressBar pct={crawlProgress.char_pct} color="bg-emerald-400" />
            <div className="flex justify-between text-xs text-gray-400">
              <span>전체 이미지 진행률</span>
              <span className="tabular-nums">{crawlProgress.total_pct.toFixed(1)}%</span>
            </div>
            <ProgressBar pct={crawlProgress.total_pct} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-[11px] text-gray-400">
            <span>완료 캐릭터: <b className="text-gray-200">{crawlProgress.completed_characters}</b></span>
            <span>포함: <b className="text-green-300">{crawlProgress.included}</b></span>
            <span>스킵: <b className="text-gray-300">{crawlProgress.skipped}</b></span>
            <span>기준 미달: <b className="text-yellow-300">{crawlProgress.below_threshold}</b></span>
            <span>속도: <b className="text-gray-200">{crawlProgress.speed_img_s.toFixed(2)} img/s</b></span>
          </div>
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-gray-800 bg-gray-950/40 p-6 text-center text-sm text-gray-500">
          크롤링을 시작하면 캐릭터별/전체 진행률이 여기에 표시됩니다.
        </div>
      )}

      {crawlHealth && (
        <div className="rounded-lg bg-gray-950 border border-gray-800 p-3 text-[11px] text-gray-400">
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            <span>
              Heartbeat:{" "}
              <b className={crawlHealth.heartbeat_ok ? "text-green-300" : "text-red-300"}>
                {crawlHealth.heartbeat_ok ? "정상" : "지연"}
              </b>
              {crawlHealth.last_event_age_sec != null ? ` (${crawlHealth.last_event_age_sec}s 전)` : ""}
            </span>
            <span>API 요청: <b className="text-gray-200">{crawlHealth.crawler?.api_requests ?? 0}</b></span>
            <span>API 오류: <b className="text-yellow-300">{crawlHealth.crawler?.api_errors ?? 0}</b></span>
            <span>다운로드 오류: <b className="text-yellow-300">{crawlHealth.crawler?.download_errors ?? 0}</b></span>
            <span>계정: <b className="text-gray-200">{crawlHealth.crawler?.account_mode ?? "unknown"}</b></span>
            {crawlHealth.remote && (
              <span>
                Remote:{" "}
                <b className={crawlHealth.remote.ok && !crawlHealth.remote.rate_limited ? "text-green-300" : "text-red-300"}>
                  {crawlHealth.remote.rate_limited ? "rate limited" : crawlHealth.remote.ok ? "ok" : "fail"}
                </b>
                {crawlHealth.remote.status_code ? ` HTTP ${crawlHealth.remote.status_code}` : ""}
                {crawlHealth.remote.latency_ms != null ? ` · ${crawlHealth.remote.latency_ms}ms` : ""}
              </span>
            )}
          </div>
          {(crawlHealth.crawler?.last_error || crawlHealth.remote?.error) && (
            <p className="mt-2 text-red-300 break-all">
              {crawlHealth.crawler?.last_error || crawlHealth.remote?.error}
            </p>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* ── 헤더 ── */}
	      <div className="flex items-center justify-between">
	        <div>
	          <h1 className="text-xl font-bold text-white">{t("common.crawl")}</h1>
	          <p className="text-sm text-gray-400 mt-0.5">{t("crawl.subtitle")}</p>
	        </div>
	        <StatusBadge state={crawlState} />
	      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* ── 캐릭터 관리 ── */}
        <div className="col-span-2 card space-y-3">
          {discovery && discovery.missing.length > 0 && (
            <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 flex items-start justify-between gap-3">
              <div>
                <p className="text-xs font-semibold text-amber-300">
                  dataset/raw에 등록되지 않은 데이터셋 {discovery.missing.length}개 발견
                </p>
                <p className="text-[11px] text-amber-100/70 mt-1">
                  이미지 파일은 삭제되지 않았습니다. 폴더명을 Danbooru 태그로 사용해 characters.json에 다시 등록할 수 있습니다.
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

          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-200">{t("crawl.char_label")}</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500">
                {selected.size > 0 ? `${selected.size}${t("dataset.selected_count", { count: selected.size })} /` : ""} 전체 {characters.length}개
              </span>
              <button
                onClick={toggleVisible}
                className="text-xs text-brand-400 hover:text-brand-300"
              >
                {visibleAllSelected ? t("crawl.deselect_all") : t("crawl.select_all")}
              </button>
              <button
                onClick={() => { setAddOpen((o) => !o); setAddErr(""); }}
                className="text-xs px-2 py-1 rounded bg-brand-600 hover:bg-brand-500 text-white transition-colors"
              >
                {t("crawl.add_btn")}
              </button>
            </div>
          </div>

          {/* 추가 폼 */}
           {addOpen && (
             <div className="bg-gray-800 rounded-lg p-3 space-y-2 border border-gray-700">
               <p className="text-xs font-medium text-gray-300">{t("crawl.new_char")}</p>
               <div className="grid grid-cols-3 gap-2">
                 <div>
                   <label className="label-text">{t("crawl.key_label")}</label>
                   <input
                     value={newKey}
                     onChange={(e) => setNewKey(e.target.value)}
                     className="input text-xs"
                     placeholder="tokino_sora"
                   />
                   <p className="text-[10px] text-gray-600 mt-0.5">영문·숫자·_ 권장</p>
                 </div>
                 <div>
                   <label className="label-text">{t("crawl.name_label")}</label>
                   <input
                     value={newName}
                     onChange={(e) => setNewName(e.target.value)}
                     className="input text-xs"
                     placeholder="Tokino Sora (선택)"
                   />
                 </div>
                 <div>
                   <label className="label-text">{t("crawl.tag_label")}</label>
                   <TagSearchInput
                     value={newTag}
                     onChange={setNewTag}
                     onPostCount={setNewPostCount}
                     onSelect={handleTagSelect}
                     placeholder={t("crawl.tag_placeholder")}
                   />
                   {newPostCount != null && (
                     <p className="text-[10px] text-green-500 mt-0.5">
                       {t("crawl.confirmed_count", { count: newPostCount.toLocaleString() })}
                     </p>
                   )}
                 </div>
               </div>
               {addErr && <p className="text-xs text-red-400">{addErr}</p>}
               <div className="flex gap-2">
                 <button onClick={handleAdd} className="btn-primary text-xs px-3 py-1">{t("common.add")}</button>
                 <button
                   onClick={() => { setAddOpen(false); setAddErr(""); setNewPostCount(null); }}
                   className="text-xs px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
                 >
                   {t("common.cancel")}
                 </button>
               </div>
	             </div>
	           )}

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-gray-800 bg-gray-950/40 p-3 space-y-2">
              <p className="text-xs font-medium text-gray-300">캐릭터 검색/필터</p>
              <input
                value={charSearch}
                onChange={(e) => setCharSearch(e.target.value)}
                className="input text-xs"
                placeholder="key, 표시 이름, Danbooru 태그 검색"
              />
              <select
                value={charFilter}
                onChange={(e) => setCharFilter(e.target.value as typeof charFilter)}
                className="input text-xs"
              >
                <option value="all">전체</option>
                <option value="selected">선택된 캐릭터</option>
                <option value="under_min">최소 이미지 수 미달</option>
                <option value="others">others 이미지 있음</option>
              </select>
              <p className="text-[11px] text-gray-500">
                표시 {filteredCharacters.length}개 / 전체 {characters.length}개
              </p>
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-950/40 p-3 space-y-2">
              <p className="text-xs font-medium text-gray-300">장르로 일괄 추가</p>
              <div className="flex gap-2">
                <input
                  value={genreQuery}
                  onChange={(e) => setGenreQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void searchGenreCharacters();
                    }
                  }}
                  className="input text-xs"
                  placeholder="예: blue archive"
                  disabled={genreLoading}
                />
                <button
                  onClick={searchGenreCharacters}
                  disabled={genreLoading || genreQuery.trim().length < 2}
                  className="btn-ghost text-xs px-3 py-1 shrink-0"
                >
                  {genreLoading ? "검색중" : "검색"}
                </button>
              </div>
              {genreErr && <p className="text-[11px] text-red-400">{genreErr}</p>}
              {genreNotice && <p className="text-[11px] text-green-400">{genreNotice}</p>}
              {genreCandidates.length > 0 && (
                <div className="rounded-md border border-gray-800 bg-gray-900">
                  <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-gray-800">
                    <p className="text-xs text-gray-300">
                      {genreSelected.size} / {genreCandidates.length}개 캐릭터를 추가하겠습니까?
                    </p>
                    <div className="flex gap-2">
                      <button onClick={toggleAllGenreCandidates} className="text-[11px] text-brand-300 hover:text-brand-200">
                        전체 토글
                      </button>
                      <button
                        onClick={importGenreCharacters}
                        disabled={genreSelected.size === 0}
                        className="text-[11px] px-2 py-1 rounded bg-brand-600 text-white disabled:opacity-40"
                      >
                        선택 추가
                      </button>
                    </div>
                  </div>
                  <details className="group" open>
                    <summary className="cursor-pointer px-3 py-2 text-[11px] text-gray-400 hover:text-gray-200">
                      후보 리스트 열기/닫기
                    </summary>
                    <div className="max-h-52 overflow-y-auto border-t border-gray-800">
                      {genreCandidates.map((candidate) => {
                        const exists = characters.some((c) => c.key === candidate.key);
                        return (
                          <label
                            key={candidate.key}
                            className={`flex items-center gap-2 px-3 py-1.5 text-xs border-b border-gray-800/60 ${
                              exists ? "text-gray-600" : "text-gray-300 hover:bg-gray-800/70"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={genreSelected.has(candidate.key)}
                              disabled={exists}
                              onChange={() => toggleGenreCandidate(candidate.key)}
                              className="accent-brand-500"
                            />
                            <span className="min-w-0 flex-1">
                              <span className="block truncate">{candidate.display_name}</span>
                              <span className="block truncate font-mono text-[10px] text-gray-500">{candidate.tag}</span>
                            </span>
                            <span className="shrink-0 text-[10px] text-gray-500">
                              {exists ? "등록됨" : `${candidate.post_count.toLocaleString()}장`}
                            </span>
                          </label>
                        );
                      })}
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>

          {/* 캐릭터 테이블 */}
          <div className="max-h-72 overflow-y-auto rounded-lg border border-gray-700">
	             {characters.length === 0 ? (
	               <p className="text-xs text-gray-500 text-center py-8">
	                 {t("crawl.empty_chars")}
	               </p>
	             ) : filteredCharacters.length === 0 ? (
	               <p className="text-xs text-gray-500 text-center py-8">
	                 검색/필터 조건에 맞는 캐릭터가 없습니다.
	               </p>
	             ) : (

               <table className="w-full text-xs">
                 <thead className="sticky top-0 bg-gray-800 text-gray-400">
                   <tr>
                     <th className="w-8 px-2 py-2"></th>
                     <th className="text-left px-2 py-2">{t("crawl.table_key")}</th>
                     <th className="text-left px-2 py-2">{t("crawl.table_name")}</th>
                     <th className="text-left px-2 py-2">{t("crawl.table_tag")}</th>
                     <th className="text-right px-2 py-2">{t("crawl.table_images")}</th>
                     <th className="w-16 px-2 py-2"></th>
                   </tr>
                 </thead>

                <tbody>
	                  {filteredCharacters.map((c) => {
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
                        <td className="px-2 py-1.5 text-right">
                          <span className="text-gray-500">{c.count}</span>
                          {(c.other_count ?? 0) > 0 && (
                            <span className="block text-[10px] text-yellow-500">
                              others {c.other_count}
                            </span>
                          )}
                        </td>
                        <td className="px-2 py-1.5 text-right">
                         {isEditing ? (
                           <div className="flex gap-1 justify-end">
                             <button
                               onClick={handleEditSave}
                               className="text-brand-400 hover:text-brand-300"
                             >
                               {t("common.save")}
                             </button>
                             <button
                               onClick={() => setEditing(null)}
                               className="text-gray-500 hover:text-gray-300"
                             >
                               {t("common.cancel")}
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
           <p className="text-sm font-medium text-gray-200">{t("crawl.config_title")}</p>
 
           <div>
             <label className="label-text">{t("crawl.min_images")}</label>
             <input
               type="number" value={minImages}
               onChange={(e) => setMinImages(+e.target.value)}
               className="input" min={1} disabled={running}
             />
           </div>
           <div>
             <label className="label-text">{t("crawl.max_images")}</label>
             <input
               type="number" value={maxImages}
               onChange={(e) => setMaxImages(+e.target.value)}
               className="input" min={1} disabled={running}
             />
           </div>
           <div>
             <label className="label-text">{t("crawl.workers")}</label>
             <input
               type="number" value={workers}
               onChange={(e) => setWorkers(+e.target.value)}
               className="input" min={1} max={16} disabled={running}
             />
           </div>
 
           <hr className="border-gray-700" />
 
           <div>
             <label className="label-text">{t("crawl.danbooru_id")}</label>
             <input
               value={username} onChange={(e) => setUsername(e.target.value)}
               className="input" placeholder={t("crawl.env_loaded")} disabled={running}
             />
           </div>
           <div>
             <label className="label-text">{t("crawl.api_key")}</label>
             <input
               type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
               className="input" placeholder={t("crawl.env_loaded")} disabled={running}
             />
           </div>
 
           <p className="text-xs text-gray-500">
             {selected.size === 0
               ? t("crawl.selection_hint")
               : t("crawl.selection_count", { count: selected.size })}
           </p>
 
           <div className="pt-1 space-y-2">
             {!running ? (
               <button
                 onClick={startCrawl}
                 disabled={characters.length === 0}
                 className="btn-primary w-full disabled:opacity-40"
               >
                 {t("crawl.start_btn")}
               </button>
             ) : (
               <button onClick={stopCrawl} className="btn-danger w-full">
                 {t("crawl.stop_btn")}
               </button>
             )}
           </div>
         </div>

      </div>

      {crawlerProgressPanel}

       {/* ── 로그 ── */}
       <div className="card">
	         <JobConsole
	           title={t("crawl.log_title")}
	           jobPath="/crawl/logs"
	           onState={handleCrawlState}
             onCrawlProgress={handleCrawlProgress}
             onCrawlHealth={setCrawlHealth}
	         />
       </div>

    </div>
  );
}
