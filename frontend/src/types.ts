// ── 공통 ────────────────────────────────────────────────
export type JobState = "idle" | "running" | "done" | "failed";

export interface WsMessage {
  type: "log" | "state" | "metric" | "progress" | "crawl_progress" | "crawl_health";
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

export interface GenreCharacterCandidate {
  key:          string;
  tag:          string;
  display_name: string;
  post_count:   number;
  source?:      string | null;
}

export interface GenreCharactersResponse {
  query:      string;
  normalized: string;
  characters: GenreCharacterCandidate[];
}

export interface DatasetCharacterCandidate {
  key:          string;
  tag:          string;
  display_name: string;
  count:        number;
}

export interface DatasetDiscovery {
  registered:     number;
  dataset_labels: number;
  missing:        DatasetCharacterCandidate[];
}

export interface CrawlRateHealth {
  account_mode:     "anonymous" | "authenticated" | string;
  api_requests:     number;
  api_errors:       number;
  download_errors:  number;
  rate_limited:     boolean;
  last_status:      number | null;
  last_error:       string | null;
  retry_after:      string | null;
  last_api_at:      number | null;
  last_latency_ms:  number | null;
  resized_images?:  number;
  resize_saved_bytes?: number;
}

export interface CrawlProgress {
  current_character:   string;
  current_index:       number;
  total_characters:    number;
  phase:               "starting" | "collecting" | "downloading" | "done" | string;
  status?:             string | null;
  char_downloaded:     number;
  char_target:         number;
  char_queue:          number;
  char_pct:            number;
  char_eta_sec:        number;
  completed_characters:number;
  included:            number;
  skipped:             number;
  below_threshold:     number;
  total_downloaded:    number;
  total_target:        number;
  total_pct:           number;
  overall_eta_sec:     number;
  speed_img_s:         number;
  health:              CrawlRateHealth;
  updated_at:          number;
}

export interface CrawlStatus extends JobStatus {
  current_progress:  CrawlProgress | null;
  health:            CrawlRateHealth | null;
  last_event_age_sec: number | null;
}

export interface CrawlHealthResponse {
  state:               JobState;
  heartbeat_ok:        boolean;
  last_event_age_sec:  number | null;
  crawler:             CrawlRateHealth | null;
  current_progress:    CrawlProgress | null;
  remote?: {
    checked:      boolean;
    ok:           boolean;
    status_code:  number | null;
    rate_limited: boolean;
    retry_after:  string | null;
    latency_ms:   number | null;
    error:        string | null;
  };
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

export type ImageSort = "name_asc" | "name_desc" | "newest" | "oldest";

export interface LargeImageItem {
  id:      string;
  label:   string;
  name:    string;
  bytes:   number;
  size_mb: number;
  width:   number | null;
  height:  number | null;
}

export interface LargeImageScan {
  label:           string | null;
  threshold_mb:    number;
  threshold_bytes: number;
  total_count:     number;
  large_count:     number;
  large_bytes:     number;
  largest:         LargeImageItem[];
}

export interface ImagePreprocessItem extends LargeImageItem {
  processed:    boolean;
  before_bytes: number;
  after_bytes:  number;
  saved_bytes:  number;
  reason:       string;
}

export interface ImagePreprocessResult {
  label:        string | null;
  threshold_mb: number;
  max_side:     number;
  quality:      number;
  scanned:      number;
  processed:    number;
  skipped:      number;
  before_bytes: number;
  after_bytes:  number;
  saved_bytes:  number;
  items:        ImagePreprocessItem[];
}

export interface ImagePreprocessStatus {
  name:          string;
  state:         JobState;
  label:         string | null;
  threshold_mb:  number;
  max_side:      number;
  quality:       number;
  dry_run:       boolean;
  total:         number;
  current:       number;
  pct:           number;
  current_image: string | null;
  processed:     number;
  skipped:       number;
  before_bytes:  number;
  after_bytes:   number;
  saved_bytes:   number;
  started_at:    number | null;
  finished_at:   number | null;
  elapsed_sec:   number | null;
  error:         string | null;
  result:        ImagePreprocessResult | null;
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
  avg_loss?:    number;   // current split running average
  avg_acc?:     number;   // current split running average
}

export interface TrainingStatus extends JobStatus {
  current_phase:    number;
  best_val_acc:     number;
  epoch_count:      number;
  phase1_epochs:    number;
  phase2_epochs:    number;
  current_progress: TrainProgress | null;
  last_metric:      TrainMetric | null;
  elapsed_sec:      number | null;
}

export interface DeviceOption {
  key:       string;
  label:     string;
  available: boolean;
  reason:    string | null;
}

export interface TrainingDevicesResponse {
  torch_version: string;
  ipex_version:  string | null;
  devices:       DeviceOption[];
}

export type TrainingMode = "fresh" | "resume" | "finetune";

export interface TrainingArtifactEntry {
  exists:   boolean;
  filename: string;
  size_mb:  number | null;
  mtime:    number | null;
}

export interface TrainingArtifacts {
  best_model:          TrainingArtifactEntry;
  checkpoint:          TrainingArtifactEntry;
  config_best_val_acc: number | null;
  config_test_acc:     number | null;
  num_classes:         number | null;
  config_backbone:     string | null;
}

export interface BackboneOption {
  key:         string;
  label:       string;
  params_m:    number;
  description: string;
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

export interface InferenceModelInfo {
  fp32_available:     boolean;
  fp16_available:     boolean;
  int8_available?:    boolean;
  int4_available?:    boolean;
  int2_available?:    boolean;
  onnx_available:     boolean;
  num_classes:        number | null;
  best_val_acc:       number | null;
  test_acc:           number | null;
  preferred_backend?: string | null;
  loaded_backend?:    string | null;
  model_ready?:       boolean;
}

export interface QuantMetrics {
  format: string;
  fp32_mb: number;
  out_mb: number;
  ratio_pct: number;
}
