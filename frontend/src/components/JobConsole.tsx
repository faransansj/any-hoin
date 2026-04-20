/**
 * JobConsole — WebSocket 로그 스트리밍 콘솔
 * jobPath: "/crawl/logs" | "/training/logs" | "/export/logs/fp16" 등
 * onState: state 변경 콜백
 * onMetric: metric 이벤트 콜백 (학습 전용)
 */
import { useEffect, useRef, useState } from "react";
import type { JobState, TrainMetric, TrainProgress } from "../types";
import { api } from "../api";

function ClipboardIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
    </svg>
  );
}

interface Props {
  jobPath:     string;
  /** 제목 텍스트. 제공 시 복사/지우기 버튼이 같은 행에 표시됨. */
  title?:      string;
  onState?:    (s: JobState) => void;
  onMetric?:   (m: TrainMetric) => void;
  onProgress?: (p: TrainProgress) => void;
  maxLines?:   number;
}

export default function JobConsole({
  jobPath,
  title,
  onState,
  onMetric,
  onProgress,
  maxLines = 400,
}: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [copied, setCopied] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const wsRef     = useRef<WebSocket | null>(null);

  function handleCopy() {
    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  async function handleClear() {
    setLines([]);
    // jobPath "/crawl/logs" → "/crawl/logs/clear" 형태로 변환
    const clearPath = jobPath.replace(/^\//, "").replace("/logs", "/logs/clear");
    try { await api.post(clearPath, {}); } catch { /* 무시 */ }
  }

  useEffect(() => {
    let ws: WebSocket;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let dead = false;

    function connect() {
      if (dead) return;
      ws = api.ws(jobPath);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data as string);
          if (msg.type === "log") {
            setLines((prev) => {
              const next = [...prev, msg.data as string];
              return next.length > maxLines ? next.slice(-maxLines) : next;
            });
          } else if (msg.type === "state" && onState) {
            onState(msg.data as JobState);
          } else if (msg.type === "metric" && onMetric) {
            onMetric(msg.data as TrainMetric);
          } else if (msg.type === "progress" && onProgress) {
            onProgress(msg.data as TrainProgress);
          }
        } catch {
          /* 무시 */
        }
      };

      ws.onclose = () => {
        if (!dead) retryTimer = setTimeout(connect, 2000);
      };
    }

    setLines([]);
    connect();

    return () => {
      dead = true;
      if (retryTimer) clearTimeout(retryTimer);
      ws?.close();
    };
  }, [jobPath]); // eslint-disable-line react-hooks/exhaustive-deps

  // 자동 스크롤
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  const buttons = (
    <div className="flex gap-1.5">
      <button
        onClick={handleCopy}
        disabled={lines.length === 0}
        title="클립보드에 복사"
        className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        <ClipboardIcon />
        {copied ? "복사됨" : "복사"}
      </button>
      <button
        onClick={handleClear}
        disabled={lines.length === 0}
        title="로그 지우기"
        className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-gray-800 text-gray-400 hover:bg-red-900 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
      >
        <TrashIcon />
        지우기
      </button>
    </div>
  );

  return (
    <div className="flex flex-col gap-2">
      {title !== undefined && (
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-gray-200">{title}</p>
          {buttons}
        </div>
      )}
      <div className="bg-gray-950 border border-gray-800 rounded-lg h-64 overflow-y-auto p-3 font-mono text-xs text-gray-300">
        {lines.length === 0 && (
          <span className="text-gray-600 italic">로그 대기 중...</span>
        )}
        {lines.map((line, i) => (
          <div key={i} className="leading-5 whitespace-pre-wrap break-all">
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
