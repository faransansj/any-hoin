import { useEffect, useRef } from "react";
import { api } from "../api";
import { useJobStore } from "../store/jobStore";
import type { JobState, TrainMetric, TrainingStatus } from "../types";

function fmtAcc(acc: number | null | undefined): string {
  return typeof acc === "number" && acc > 0 ? `${(acc * 100).toFixed(2)}%` : "n/a";
}

function notifyTraining(title: string, body: string) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    new Notification(title, { body });
  } catch {
    /* Browser notifications are best-effort only. */
  }
}

export default function TrainingMonitor() {
  const previousState = useRef<JobState | null>(null);
  const defaultTitle = useRef(document.title || "HoloScope Studio");
  const { setTrainState, setTrainProgress, setMetrics } = useJobStore();

  useEffect(() => {
    let dead = false;

    async function syncMetrics(force: boolean, expectedCount: number) {
      const currentCount = useJobStore.getState().trainMetrics.length;
      if (!force && currentCount === expectedCount) return;

      const res = await api.get<{ metrics: TrainMetric[] }>("/training/metrics");
      if (!dead) setMetrics(res.metrics);
    }

    async function poll() {
      try {
        const status = await api.get<TrainingStatus>("/training/status");
        if (dead) return;

        const previous = previousState.current;
        setTrainState(status.state);
        setTrainProgress(status.state === "running" ? status.current_progress : null);

        const finishedNow = previous === "running" && status.state !== "running";
        await syncMetrics(finishedNow, status.epoch_count);

        if (status.state === "running") {
          const progress = status.current_progress;
          const pct = progress ? `${progress.pct}% ${progress.split}` : "running";
          document.title = `Training ${pct} - ${defaultTitle.current}`;
        } else if (finishedNow) {
          const title = status.state === "done" ? "Training complete" : "Training failed";
          const body = `best val_acc ${fmtAcc(status.best_val_acc)}, epochs ${status.epoch_count}`;
          document.title = `${title} - ${defaultTitle.current}`;
          notifyTraining(title, body);
        } else {
          document.title = defaultTitle.current;
        }

        previousState.current = status.state;
      } catch (err) {
        console.error(err);
      }
    }

    void poll();
    const timer = setInterval(poll, 5000);
    window.addEventListener("focus", poll);

    return () => {
      dead = true;
      clearInterval(timer);
      window.removeEventListener("focus", poll);
      document.title = defaultTitle.current;
    };
  }, [setMetrics, setTrainProgress, setTrainState]);

  return null;
}
