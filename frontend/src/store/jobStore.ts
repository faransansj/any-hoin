import { create } from "zustand";
import type { JobState, TrainMetric, TrainProgress } from "../types";

interface JobStore {
  crawlState:    JobState;
  trainState:    JobState;
  quantState:    JobState;
  onnxState:     JobState;
  trainMetrics:  TrainMetric[];
  bestValAcc:    number;
  trainProgress: TrainProgress | null;

  setCrawlState:    (s: JobState) => void;
  setTrainState:    (s: JobState) => void;
  setQuantState:    (s: JobState) => void;
  setOnnxState:     (s: JobState) => void;
  pushMetric:       (m: TrainMetric) => void;
  setMetrics:       (metrics: TrainMetric[]) => void;
  resetMetrics:     () => void;
  setTrainProgress: (p: TrainProgress | null) => void;
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
  trainMetrics:  [],
  bestValAcc:    0,
  trainProgress: null,

  setCrawlState:    (s) => set({ crawlState: s }),
  setTrainState:    (s) => set({ trainState: s }),
  setQuantState:    (s) => set({ quantState: s }),
  setOnnxState:     (s) => set({ onnxState: s }),
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
  resetMetrics:     () => set({ trainMetrics: [], bestValAcc: 0, trainProgress: null }),
  setTrainProgress: (p) => set({ trainProgress: p }),
}));
