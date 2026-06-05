import type { JobState } from "../types";
import { useTranslation } from "react-i18next";

const styles: Record<JobState, string> = {
  idle:    "bg-gray-700 text-gray-300",
  running: "bg-blue-900 text-blue-300 animate-pulse",
  done:    "bg-green-900 text-green-300",
  failed:  "bg-red-900 text-red-300",
};

export default function StatusBadge({ state }: { state: JobState }) {
  const { i18n } = useTranslation();
  const isKo = i18n.resolvedLanguage?.startsWith("ko") ?? false;
  const labels: Record<JobState, string> = {
    idle:    isKo ? "대기" : "Idle",
    running: isKo ? "실행 중" : "Running",
    done:    isKo ? "완료" : "Done",
    failed:  isKo ? "실패" : "Failed",
  };

  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${styles[state]}`}>
      {labels[state]}
    </span>
  );
}
