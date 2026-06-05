import { useEffect, useRef } from "react";
import { api } from "../api";
import { useJobStore } from "../store/jobStore";
import type { JobState, TrainBottleneck, TrainMetric, TrainProgress, TrainingStatus } from "../types";

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
  const progressCursor = useRef<{ split: "train" | "val"; batch_cur: number; at: number } | null>(null);
  const slowdownSince = useRef<number | null>(null);
  const speedSamples = useRef<number[]>([]);
  const { setTrainState, setTrainProgress, setTrainBottleneck, setMetrics } = useJobStore();

  function median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  function detectBottleneck(progress: TrainProgress, now: number): TrainBottleneck | null {
    const cursor = progressCursor.current;
    const advanced = !cursor || cursor.split !== progress.split || progress.batch_cur > cursor.batch_cur;
    if (advanced) {
      progressCursor.current = { split: progress.split, batch_cur: progress.batch_cur, at: now };
      slowdownSince.current = null;
    }

    const stalledSec = progressCursor.current ? Math.max(0, (now - progressCursor.current.at) / 1000) : 0;
    const speed = Math.max(0, progress.speed_it_s);

    if (speed > 0) {
      speedSamples.current.push(speed);
      if (speedSamples.current.length > 20) speedSamples.current.shift();
    }
    const baseline = median(speedSamples.current);
    const ratio = baseline > 0 ? speed / baseline : 1;

    if (stalledSec >= 45) {
      return {
        level: "critical",
        code: "stalled",
        split: progress.split,
        stalled_sec: stalledSec,
        current_speed: speed,
        baseline_speed: baseline,
        slowdown_ratio: ratio,
        observed_at: now,
      };
    }

    if (baseline >= 0.5 && ratio < 0.35) {
      if (slowdownSince.current == null) slowdownSince.current = now;
      const slowdownSec = (now - slowdownSince.current) / 1000;
      if (slowdownSec >= 20) {
        return {
          level: "warning",
          code: "slowdown",
          split: progress.split,
          stalled_sec: stalledSec,
          current_speed: speed,
          baseline_speed: baseline,
          slowdown_ratio: ratio,
          observed_at: now,
        };
      }
    } else {
      slowdownSince.current = null;
    }

    return null;
  }

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
        const progress = status.state === "running" ? status.current_progress : null;
        setTrainProgress(progress);

        const finishedNow = previous === "running" && status.state !== "running";
        await syncMetrics(finishedNow, status.epoch_count);

        if (status.state === "running") {
          const pct = progress ? `${progress.pct}% ${progress.split}` : "running";
          document.title = `Training ${pct} - ${defaultTitle.current}`;
          if (progress) {
            setTrainBottleneck(detectBottleneck(progress, Date.now()));
          } else {
            setTrainBottleneck(null);
          }
        } else if (finishedNow) {
          const title = status.state === "done" ? "Training complete" : "Training failed";
          const body = `best val_acc ${fmtAcc(status.best_val_acc)}, epochs ${status.epoch_count}`;
          document.title = `${title} - ${defaultTitle.current}`;
          notifyTraining(title, body);
          setTrainBottleneck(null);
          progressCursor.current = null;
          slowdownSince.current = null;
          speedSamples.current = [];
        } else {
          document.title = defaultTitle.current;
          setTrainBottleneck(null);
          progressCursor.current = null;
          slowdownSince.current = null;
          speedSamples.current = [];
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
  }, [setMetrics, setTrainBottleneck, setTrainProgress, setTrainState]);

  return null;
}
