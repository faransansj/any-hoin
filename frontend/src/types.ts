// ── 공통 ────────────────────────────────────────────────
export type JobState = "idle" | "running" | "done" | "failed";

export interface WsMessage {
  type: "log" | "state" | "metric" | "progress";
  data: unknown;
}

// ── 캐릭터 (범용) ────────────────────────────────────────
export interface Character {
  key:          string;   // 폴더명 = 클래스명
  tag:          string;   // Danbooru 검색 태그
  display_name: string;   // UI 표시 이름
  count:        number;   // 현재 이미지 수
  other_count?: number;   // others/<key>에 격리된 이미지 수
  total_count?: number;   // count + other_count
}

// ── 라벨 ────────────────────────────────────────────────
export interface Label {
  name:    string;
  count:   number;
  warning: boolean;
}

// ── 이미지 ──────────────────────────────────────────────
export interface ImageItem {
  id:        string;   // "label/filename"
  name:      string;
  label:     string;
  url:       string;
  thumbnail: string;
}

// ── 학습 메트릭 ─────────────────────────────────────────
export interface TrainMetric {
  epoch:        number;
  total_epochs: number;
  phase:        1 | 2;
  train_loss:   number;
  train_acc:    number;
  val_loss:     number;
  val_acc:      number;
}

// ── 학습 진행률 (배치 단위) ──────────────────────────────
export interface TrainProgress {
  split:       "train" | "val";
  pct:         number;   // 0-100
  batch_cur:   number;
  batch_total: number;
  eta_sec:     number;   // -1 = unknown
  speed_it_s:  number;   // batches/sec
}

// ── 잡 상태 ────────────────────────────────────────────
export interface JobStatus {
  name:  string;
  state: JobState;
}

export type QuantFormat = "fp16" | "int8" | "int4" | "int2";

export interface ExportStatus {
  quant: JobStatus;
  onnx:  JobStatus;
}

// ── 모델 정보 ───────────────────────────────────────────
export interface ModelEntry {
  exists:   boolean;
  size_mb:  number | null;
  filename: string;
}

export interface ModelMap {
  fp32: ModelEntry;
  fp16: ModelEntry;
  int8: ModelEntry;
  int4: ModelEntry;
  int2: ModelEntry;
  onnx: ModelEntry;
}

export interface ModelsResponse {
  models: ModelMap;
  config_acc: number | null;
}

export interface QuantMetrics {
  format: string;
  fp32_mb: number;
  out_mb: number;
  ratio_pct: number;
}
