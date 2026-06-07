import { create } from "zustand";
import type { JobState, TrainBottleneck, TrainMetric, TrainProgress } from "../types";

interface JobStore {
  crawlState:    JobState;
  trainState:    JobState;
  quantState:    JobState;
  onnxState:     JobState;
  hoinState:     JobState;
  trainMetrics:  TrainMetric[];
  bestValAcc:    number;
  trainProgress: TrainProgress | null;
  trainBottleneck: TrainBottleneck | null;

  setCrawlState:    (s: JobState) => void;
  setTrainState:    (s: JobState) => void;
  setQuantState:    (s: JobState) => void;
  setOnnxState:     (s: JobState) => void;
  setHoinState:     (s: JobState) => void;
  pushMetric:       (m: TrainMetric) => void;
  setMetrics:       (metrics: TrainMetric[]) => void;
  resetMetrics:     () => void;
  setTrainProgress: (p: TrainProgress | null) => void;
  setTrainBottleneck: (b: TrainBottleneck | null) => void;
}

function metricKey(m: TrainMetric): string {
  return `${m.phase}:${m.epoch}:${m.total_epochs}`;
}

function bestAcc(metrics: TrainMetric[]): number {
  return metrics.reduce((best, m) => Math.max(best, m.val_acc), 0);
}

export const useJobStore = create<JobStore>((set) => ({
  crawlState:    "idle",
  trainState:    "idle",
  quantState:    "idle",
  onnxState:     "idle",
  hoinState:     "idle",
  trainMetrics:  [],
  bestValAcc:    0,
  trainProgress: null,
  trainBottleneck: null,

  setCrawlState:    (s) => set({ crawlState: s }),
  setTrainState:    (s) => set({ trainState: s }),
  setQuantState:    (s) => set({ quantState: s }),
  setOnnxState:     (s) => set({ onnxState: s }),
  setHoinState:     (s) => set({ hoinState: s }),
  pushMetric:       (m) => set((st) => {
    const next = st.trainMetrics.filter((metric) => metricKey(metric) !== metricKey(m));
    next.push(m);
    return {
      trainMetrics: next,
      bestValAcc:   Math.max(st.bestValAcc, m.val_acc),
    };
  }),
  setMetrics:       (metrics) => set({
    trainMetrics: metrics,
    bestValAcc: bestAcc(metrics),
  }),
  resetMetrics:     () => set({ trainMetrics: [], bestValAcc: 0, trainProgress: null, trainBottleneck: null }),
  setTrainProgress: (p) => set({ trainProgress: p }),
  setTrainBottleneck: (b) => set({ trainBottleneck: b }),
}));
