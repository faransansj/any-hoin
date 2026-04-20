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
  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
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
      onPostCount?.(null);
      return;
    }
    timerRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await api.get<{ tags: TagSuggestion[] }>(
          `/crawl/tags/search?q=${encodeURIComponent(q)}&limit=10`
        );
        setSuggestions(r.tags);
        setOpen(r.tags.length > 0);
        setCursor(-1);
      } finally {
        setLoading(false);
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

      {open && (
        <ul
          ref={listRef}
          className="absolute z-50 mt-0.5 w-full bg-gray-900 border border-gray-700 rounded-md shadow-xl max-h-52 overflow-y-auto text-xs"
        >
          {suggestions.map((s, i) => (
            <li
              key={s.name}
              onMouseDown={() => select(s)}
              onMouseEnter={() => setCursor(i)}
              className={`flex items-center justify-between px-3 py-1.5 cursor-pointer transition-colors ${
                i === cursor ? "bg-brand-600/40 text-white" : "hover:bg-gray-800 text-gray-300"
              }`}
            >
              <span className="font-mono truncate">{s.name}</span>
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
