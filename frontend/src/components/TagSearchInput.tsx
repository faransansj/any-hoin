/**
 * Danbooru 태그 자동완성 입력 컴포넌트
 * - 350ms 디바운스 검색
 * - 포스트 수 표시
 * - 키보드 탐색 (↑ ↓ Enter Esc)
 */
import { useEffect, useRef, useState } from "react";
import { api } from "../api";

interface TagSuggestion {
  name: string;
  post_count: number;
  label?: string | null;
  antecedent?: string | null;
  source?: string | null;
}

interface Props {
  value: string;
  onChange: (value: string) => void;
  onPostCount?: (count: number | null) => void;
  onSelect?: (tag: string, postCount: number) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export default function TagSearchInput({
  value, onChange, onPostCount, onSelect, placeholder, disabled, className,
}: Props) {
  const [suggestions, setSuggestions] = useState<TagSuggestion[]>([]);
  const [loading,     setLoading]     = useState(false);
  const [open,        setOpen]        = useState(false);
  const [cursor,      setCursor]      = useState(-1);
  const [error,       setError]       = useState("");
  const [searched,    setSearched]    = useState(false);
  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestRef = useRef(0);
  const wrapRef   = useRef<HTMLDivElement>(null);
  const listRef   = useRef<HTMLUListElement>(null);

  // 외부 클릭 시 닫기
  useEffect(() => {
    function onDown(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  // 커서 위치가 바뀌면 해당 항목이 보이도록 스크롤
  useEffect(() => {
    if (cursor < 0 || !listRef.current) return;
    const item = listRef.current.children[cursor] as HTMLElement | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  function triggerSearch(q: string) {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (q.length < 2) {
      setSuggestions([]);
      setOpen(false);
      setError("");
      setSearched(false);
      onPostCount?.(null);
      return;
    }
    timerRef.current = setTimeout(async () => {
      const requestId = ++requestRef.current;
      setLoading(true);
      setError("");
      try {
        const r = await api.get<{ tags: TagSuggestion[]; error?: string }>(
          `/crawl/tags/search?q=${encodeURIComponent(q)}&limit=10`
        );
        if (requestId !== requestRef.current) return;
        setSuggestions(r.tags);
        setSearched(true);
        setOpen(true);
        setCursor(-1);
      } catch (e: any) {
        if (requestId !== requestRef.current) return;
        setSuggestions([]);
        setSearched(true);
        setError(e?.message ?? "검색 실패");
        setOpen(true);
      } finally {
        if (requestId === requestRef.current) setLoading(false);
      }
    }, 350);
  }

  function handleChange(v: string) {
    onChange(v);
    triggerSearch(v);
  }

  function select(s: TagSuggestion) {
    onChange(s.name);
    onPostCount?.(s.post_count);
    onSelect?.(s.name, s.post_count);
    setSuggestions([]);
    setError("");
    setSearched(false);
    setOpen(false);
    setCursor(-1);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, suggestions.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === "Enter" && cursor >= 0) {
      e.preventDefault();
      select(suggestions[cursor]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={wrapRef} className="relative">
      <div className="relative">
        <input
          value={value}
          onChange={(e) => handleChange(e.target.value)}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className={className ?? "input text-xs w-full"}
          autoComplete="off"
          spellCheck={false}
        />
        {loading && (
          <span className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 text-[10px] animate-pulse">
            검색중
          </span>
        )}
      </div>

      {open && (suggestions.length > 0 || error || searched) && (
        <ul
          ref={listRef}
          className="absolute z-[100] mt-0.5 w-full min-w-72 bg-gray-900 border border-gray-700 rounded-md shadow-xl max-h-64 overflow-y-auto text-xs"
        >
          {error && (
            <li className="px-3 py-2 text-red-300 bg-red-950/40">
              Danbooru 태그 검색 실패: {error}
            </li>
          )}
          {!error && searched && suggestions.length === 0 && (
            <li className="px-3 py-2 text-gray-500">
              검색 결과 없음
            </li>
          )}
          {suggestions.map((s, i) => (
            <li
              key={s.name}
              onMouseDown={() => select(s)}
              onMouseEnter={() => setCursor(i)}
              className={`flex items-center justify-between gap-3 px-3 py-1.5 cursor-pointer transition-colors ${
                i === cursor ? "bg-brand-600/40 text-white" : "hover:bg-gray-800 text-gray-300"
              }`}
            >
              <span className="min-w-0">
                <span className="block font-mono truncate">{s.name}</span>
                {(s.antecedent || (s.label && s.label !== s.name)) && (
                  <span className="block truncate text-[10px] text-gray-500">
                    {s.antecedent ? `alias: ${s.antecedent}` : s.label}
                  </span>
                )}
              </span>
              <span className="ml-3 shrink-0 text-gray-500">
                {s.post_count.toLocaleString()}장
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
