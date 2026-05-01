/**
 * Training 페이지
 * - 학습 파라미터 설정
 * - 실시간 진행률 (배치 / 에포크 / 페이즈 / ETA)
 * - 메트릭 차트 + 로그
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import { useTranslation } from "react-i18next";
import JobConsole from "../components/JobConsole";
import MetricsChart from "../components/MetricsChart";
import StatusBadge from "../components/StatusBadge";
import { useJobStore } from "../store/jobStore";
import type {
  BackboneOption,
  DeviceOption,
  TrainMetric,
  TrainProgress,
  TrainingArtifacts,
  TrainingDevicesResponse,
  TrainingMode,
  TrainingStatus,
} from "../types";

// ── 헬퍼 ────────────────────────────────────────────────

function fmtEta(sec: number): string {
  if (sec < 0) return "?";
  if (sec === 0) return "0s";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtPct(value: number | null | undefined): string {
  return value != null && value > 0 ? `${(value * 100).toFixed(2)}%` : "—";
}

interface ProgressBarProps {
  pct:   number;   // 0-100
  color?: string;  // tailwind bg class
  thin?:  boolean;
}
function ProgressBar({ pct, color = "bg-brand-500", thin }: ProgressBarProps) {
  return (
    <div className={`w-full bg-gray-700 rounded-full overflow-hidden ${thin ? "h-1" : "h-1.5"}`}>
      <div
        className={`${color} h-full rounded-full transition-all duration-300`}
        style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
      />
    </div>
  );
}

type NotificationStatus = NotificationPermission | "unsupported";

function currentNotificationPermission(): NotificationStatus {
  return "Notification" in window ? Notification.permission : "unsupported";
}

async function requestTrainingNotifications(): Promise<NotificationStatus> {
  if (!("Notification" in window)) return "unsupported";
  if (Notification.permission !== "default") return Notification.permission;
  try {
    return await Notification.requestPermission();
  } catch {
    /* 알림 권한은 학습 시작을 막지 않는다. */
    return Notification.permission;
  }
}

// ── 컴포넌트 ─────────────────────────────────────────────

const DEFAULT_DEVICE_OPTIONS: DeviceOption[] = [
  { key: "auto", label: "auto", available: true, reason: null },
  { key: "cuda", label: "CUDA", available: true, reason: null },
  { key: "mps", label: "MPS (Apple)", available: true, reason: null },
  { key: "xpu", label: "XPU (Intel Arc)", available: true, reason: null },
  { key: "cpu", label: "CPU", available: true, reason: null },
];

export default function Training() {
  const { t } = useTranslation();
  const {
    trainState, setTrainState,
    trainMetrics, bestValAcc, pushMetric, setMetrics, resetMetrics,
    trainProgress, setTrainProgress,
  } = useJobStore();

  const [batchSize,       setBatchSize]       = useState(32);
  const [phase1Epochs,    setPhase1Epochs]    = useState(5);
  const [phase2Epochs,    setPhase2Epochs]    = useState(30);
  const [phase2Lr,        setPhase2Lr]        = useState("1e-5");
  const [patience,        setPatience]        = useState(7);
  const [device,          setDevice]          = useState("auto");
  const [deviceOptions,   setDeviceOptions]   = useState<DeviceOption[]>(DEFAULT_DEVICE_OPTIONS);
  const [backbone,        setBackbone]        = useState(
    () => localStorage.getItem("training.backbone") ?? "swin_tiny_patch4_window7_224"
  );
  const [backboneOptions, setBackboneOptions] = useState<BackboneOption[]>([]);
  const [trainingMode,    setTrainingMode]    = useState<TrainingMode>("resume");
  const [artifacts,       setArtifacts]       = useState<TrainingArtifacts | null>(null);
  const [noAmp,           setNoAmp]           = useState(false);
  const [mixupAlpha,      setMixupAlpha]      = useState(
    () => localStorage.getItem("training.mixup_alpha")  ?? "0.0"
  );
  const [cutmixAlpha,     setCutmixAlpha]     = useState(
    () => localStorage.getItem("training.cutmix_alpha") ?? "0.0"
  );
  const [emaEnabled,      setEmaEnabled]      = useState(
    () => localStorage.getItem("training.ema_enabled") === "true"
  );
  const [emaDecay,        setEmaDecay]        = useState(
    () => localStorage.getItem("training.ema_decay") ?? "0.9998"
  );
  const [notificationPermission, setNotificationPermission] = useState<NotificationStatus>(
    () => currentNotificationPermission()
  );

  // ── 얼굴 크롭 전처리 ─────────────────────────────────────
  const [segState,     setSegState]     = useState<"idle"|"running"|"done"|"failed">("idle");
  const [segPct,       setSegPct]       = useState(0);
  const [segClass,     setSegClass]     = useState("");
  const [segEta,       setSegEta]       = useState(-1);
  const [segInputDir,  setSegInputDir]  = useState("./dataset/raw");
  const [segOutputDir, setSegOutputDir] = useState("./dataset/raw_seg");
  const [segBackend,   setSegBackend]   = useState<"cascade"|"yolo">("cascade");
  const [useFaceCrop,  setUseFaceCrop]  = useState(false);
  const [segLogOpen,   setSegLogOpen]   = useState(false);

  useEffect(() => {
    api.get<{ state: string; pct: number; current_class: string; eta_sec: number; output_dir: string }>(
      "/segmentation/status"
    ).then((r) => {
      setSegState(r.state as "idle"|"running"|"done"|"failed");
      setSegPct(r.pct ?? 0);
      setSegClass(r.current_class ?? "");
      setSegEta(r.eta_sec ?? -1);
      if (r.output_dir) setSegOutputDir(r.output_dir);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    api.get<TrainingStatus>("/training/status")
      .then((r) => {
        setTrainState(r.state);
        setTrainProgress(r.state === "running" ? r.current_progress : null);
        setPhase1Epochs(r.phase1_epochs);
        setPhase2Epochs(r.phase2_epochs);
      })
      .catch(console.error);

    api.get<{ metrics: TrainMetric[] }>("/training/metrics")
      .then((r) => setMetrics(r.metrics))
      .catch(console.error);

    api.get<TrainingDevicesResponse>("/training/devices")
      .then((r) => {
        setDeviceOptions(r.devices);
        setDevice((prev) => {
          const selected = r.devices.find((item) => item.key === prev);
          return selected?.available ? prev : "auto";
        });
      })
      .catch(console.error);

    api.get<TrainingArtifacts>("/training/artifacts")
      .then(setArtifacts)
      .catch(console.error);

    api.get<{ backbones: BackboneOption[]; default: string }>("/training/backbones")
      .then((r) => {
        setBackboneOptions(r.backbones);
      })
      .catch(console.error);
  }, [setMetrics, setTrainProgress, setTrainState]);

  useEffect(() => {
    if (!artifacts) return;
    setTrainingMode((current) => {
      if (current === "resume" && artifacts.checkpoint.exists) return current;
      if (current === "finetune" && artifacts.best_model.exists) return current;
      if (artifacts.checkpoint.exists) return "resume";
      if (artifacts.best_model.exists) return "finetune";
      return "fresh";
    });
    if (artifacts.config_backbone && !localStorage.getItem("training.backbone")) {
      setBackbone(artifacts.config_backbone);
    }
  }, [artifacts]);

  async function startSeg() {
    setSegPct(0);
    setSegClass("");
    await api.post("/segmentation/start", {
      input_dir:   segInputDir,
      output_dir:  segOutputDir,
      backend:     segBackend,
    });
    setSegState("running");
  }

  async function stopSeg() {
    await api.post("/segmentation/stop");
  }

  async function startTraining() {
    resetMetrics();
    setNotificationPermission(await requestTrainingNotifications());
    await api.post("/training/start", {
      batch_size:    batchSize,
      phase1_epochs: phase1Epochs,
      phase2_epochs: phase2Epochs,
      phase2_lr:     parseFloat(phase2Lr),
      face_crop_dir: useFaceCrop ? segOutputDir : "",
      patience,
      device,
      backbone,
      training_mode: trainingMode,
      finetune: trainingMode === "finetune",
      initial_best_val_acc: trainingMode === "finetune" ? artifacts?.config_best_val_acc ?? 0 : 0,
      no_amp: noAmp,
      mixup_alpha:  parseFloat(mixupAlpha)  || 0,
      cutmix_alpha: parseFloat(cutmixAlpha) || 0,
      ema_decay:    emaEnabled ? (parseFloat(emaDecay) || 0.9998) : 0,
    });
    setTrainState("running");
  }

  async function stopTraining() {
    await api.post("/training/stop");
  }

  async function enableNotifications() {
    setNotificationPermission(await requestTrainingNotifications());
  }

  const running = trainState === "running";
  const latestMetric = trainMetrics.at(-1);
  const xpuOption = deviceOptions.find((item) => item.key === "xpu");

  // ── 진행률 계산 ────────────────────────────────────────

  // 에포크 단위 진행률
  const totalEpochs   = phase1Epochs + phase2Epochs;
  const currentPhase  = latestMetric?.phase ?? 1;
  const phaseEpoch    = latestMetric?.epoch ?? 0;
  const phaseTotal    = currentPhase === 1 ? phase1Epochs : phase2Epochs;
  const overallEpoch  = currentPhase === 1
    ? phaseEpoch
    : phase1Epochs + phaseEpoch;

  const overallPct = totalEpochs > 0 ? (overallEpoch / totalEpochs) * 100 : 0;
  const phasePct   = phaseTotal  > 0 ? (phaseEpoch  / phaseTotal)  * 100 : 0;

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">{t("common.train")}</h1>
          <p className="text-sm text-gray-400 mt-0.5">{t("training.subtitle")}</p>
        </div>
        <StatusBadge state={trainState} />
      </div>

      <div className="grid grid-cols-3 gap-4">
         {/* ── 설정 패널 ── */}
         <div className="card space-y-3">
           <p className="text-sm font-medium text-gray-200">{t("training.params_title")}</p>
 
           <div>
            <label className="label-text">백본 모델</label>
            <select
              value={backbone}
              onChange={(e) => {
                setBackbone(e.target.value);
                localStorage.setItem("training.backbone", e.target.value);
              }}
              className="input"
              disabled={running}
            >
              {backboneOptions.length > 0
                ? backboneOptions.map((b) => (
                    <option key={b.key} value={b.key}>{b.label}</option>
                  ))
                : <option value={backbone}>{backbone}</option>
              }
            </select>
            {backboneOptions.length > 0 && (() => {
              const selected = backboneOptions.find((b) => b.key === backbone);
              return selected ? (
                <p className="mt-1 text-[11px] leading-4 text-gray-500">{selected.description}</p>
              ) : null;
            })()}
            {artifacts?.config_backbone && artifacts.config_backbone !== backbone && (
              <p className="mt-1 text-[11px] text-amber-400">
                현재 저장된 모델: {artifacts.config_backbone}
              </p>
            )}
          </div>

          <div>
             <label className="label-text">{t("training.device")}</label>
             <select value={device} onChange={(e) => setDevice(e.target.value)}
               className="input" disabled={running}>
              {deviceOptions.map((option) => (
                <option key={option.key} value={option.key} disabled={!option.available}>
                  {option.label}{option.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
            {xpuOption && !xpuOption.available && (
              <p className="mt-1 text-[11px] leading-4 text-amber-400">
                {xpuOption.reason}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-2">
             <div>
               <label className="label-text">{t("training.batch_size")}</label>
               <input type="number" value={batchSize} min={1}
                 onChange={(e) => setBatchSize(+e.target.value)}
                 className="input" disabled={running} />
             </div>
             <div>
               <label className="label-text">{t("training.patience")}</label>
               <input type="number" value={patience} min={0}
                 onChange={(e) => setPatience(+e.target.value)}
                 className="input" disabled={running} />
             </div>

          </div>

           <div>
             <label className="label-text">{t("training.phase1_epochs")}</label>
             <input type="number" value={phase1Epochs} min={1}
               onChange={(e) => setPhase1Epochs(+e.target.value)}
               className="input" disabled={running} />
           </div>
           <div>
             <label className="label-text">{t("training.phase2_epochs")}</label>
             <input type="number" value={phase2Epochs} min={1}
               onChange={(e) => setPhase2Epochs(+e.target.value)}
               className="input" disabled={running} />
           </div>
           <div>
             <label className="label-text">{t("training.phase2_lr")}</label>
             <input value={phase2Lr}
               onChange={(e) => setPhase2Lr(e.target.value)}
               className="input" disabled={running} />
           </div>


          <div>
            <label className="label-text">학습 시작 방식</label>
            <select
              value={trainingMode}
              onChange={(e) => setTrainingMode(e.target.value as TrainingMode)}
              className="input"
              disabled={running}
            >
              <option value="resume" disabled={!artifacts?.checkpoint.exists}>
                체크포인트 이어서
              </option>
              <option value="finetune" disabled={!artifacts?.best_model.exists}>
                기존 best_model 파인튜닝
              </option>
              <option value="fresh">처음부터 새 학습</option>
            </select>
            <p className="mt-1 text-[11px] leading-4 text-gray-500">
              {trainingMode === "resume"
                ? `checkpoint.pth에서 이어서 시작합니다. ${artifacts?.checkpoint.exists ? "" : "현재 체크포인트가 없습니다."}`
                : trainingMode === "finetune"
                  ? `best_model.pth를 로드해 Phase 2부터 이어갑니다. 기존 best ${fmtPct(artifacts?.config_best_val_acc)}`
                  : "기존 checkpoint.pth를 무시하고 새 모델로 Phase 1부터 시작합니다."}
            </p>
          </div>

          <div className="space-y-1 pt-1">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input type="checkbox" checked={noAmp} onChange={(e) => setNoAmp(e.target.checked)}
                className="accent-brand-500" disabled={running} />
              {t("training.no_amp")}
            </label>
          </div>

          {/* ── Augmentation ── */}
          <div className="border-t border-gray-800/60 pt-2 space-y-2">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-600">Augmentation Mix</p>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="label-text">
                  Mixup α <span className="text-gray-600 font-normal">(0=끄기)</span>
                </label>
                <input
                  type="number" step="0.1" min={0} max={2}
                  value={mixupAlpha}
                  onChange={(e) => { setMixupAlpha(e.target.value); localStorage.setItem("training.mixup_alpha", e.target.value); }}
                  className="input" disabled={running}
                />
              </div>
              <div>
                <label className="label-text">
                  CutMix α <span className="text-gray-600 font-normal">(0=끄기)</span>
                </label>
                <input
                  type="number" step="0.1" min={0} max={2}
                  value={cutmixAlpha}
                  onChange={(e) => { setCutmixAlpha(e.target.value); localStorage.setItem("training.cutmix_alpha", e.target.value); }}
                  className="input" disabled={running}
                />
              </div>
            </div>
            <p className="text-[11px] text-gray-600 leading-4">
              권장: Mixup 0.4 · CutMix 1.0. 둘 다 켜면 배치마다 50/50으로 선택.
            </p>
          </div>

          {/* ── EMA ── */}
          <div className="space-y-1.5">
            <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
              <input
                type="checkbox" checked={emaEnabled}
                onChange={(e) => { setEmaEnabled(e.target.checked); localStorage.setItem("training.ema_enabled", String(e.target.checked)); }}
                className="accent-brand-500" disabled={running}
              />
              EMA (Exponential Moving Average)
            </label>
            {emaEnabled && (
              <div className="pl-5">
                <label className="label-text">EMA Decay</label>
                <input
                  value={emaDecay}
                  onChange={(e) => { setEmaDecay(e.target.value); localStorage.setItem("training.ema_decay", e.target.value); }}
                  className="input" disabled={running}
                />
                <p className="mt-1 text-[11px] text-gray-600 leading-4">
                  권장 0.9998 — 클수록 느린 추적, 작을수록 빠른 추적
                </p>
              </div>
            )}
          </div>

           <div className="pt-1">
             {!running ? (
               <button onClick={startTraining} className="btn-primary w-full">{t("training.start_btn")}</button>
             ) : (
               <button onClick={stopTraining} className="btn-danger w-full">{t("training.stop_btn")}</button>
             )}
           </div>

           <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3 space-y-2">
             <div className="flex items-center justify-between gap-2">
               <p className="text-xs font-medium text-gray-300">{t("training.monitoring_title")}</p>
               <span className="text-[10px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
                 {t(`training.notification_${notificationPermission}`)}
               </span>
             </div>
             <p className="text-[11px] leading-4 text-gray-500">{t("training.monitoring_hint")}</p>
             {notificationPermission === "default" && (
               <button
                 type="button"
                 onClick={enableNotifications}
                 className="text-[11px] text-brand-300 hover:text-brand-200"
               >
                 {t("training.enable_notifications")}
               </button>
             )}
           </div>

        </div>

        {/* ── 오른쪽 패널 ── */}
        <div className="col-span-2 space-y-4">

          {/* 요약 카드 */}
          <div className="grid grid-cols-3 gap-3">
             {[
               { label: t("training.best_val_acc"), value: bestValAcc > 0 ? `${(bestValAcc * 100).toFixed(2)}%` : "—" },
               { label: t("training.epoch"),        value: latestMetric ? `${latestMetric.epoch}/${latestMetric.total_epochs}` : "—" },
               { label: t("training.phase"),        value: latestMetric ? `Phase ${latestMetric.phase}` : "—" },
             ].map(({ label, value }) => (

              <div key={label} className="card text-center">
                <p className="text-xs text-gray-500 mb-1">{label}</p>
                <p className="text-lg font-bold text-white">{value}</p>
              </div>
            ))}
          </div>

          {/* 진행률 패널 — running 또는 에포크 기록이 있을 때 표시 */}
          {(running || overallEpoch > 0) && (
            <div className="card space-y-3">

              {/* 전체 / 페이즈 진행률 */}
              <div className="space-y-2">
                 <div className="flex justify-between text-xs text-gray-400">
                   <span>{t("training.overall_progress")}</span>
                   <span className="tabular-nums">{overallEpoch} / {totalEpochs} epoch</span>
                 </div>
                 <ProgressBar pct={overallPct} color="bg-brand-500" />
 
                 <div className="flex justify-between text-xs text-gray-500">
                   <span>{t("training.phase_progress", { phase: currentPhase })}</span>
                   <span className="tabular-nums">{phaseEpoch} / {phaseTotal} epoch</span>
                 </div>
                 <ProgressBar pct={phasePct} color="bg-indigo-500" thin />

              </div>

              {/* 현재 배치 진행률 */}
              {trainProgress && running && (
                <>
                  <div className="border-t border-gray-700/60 pt-3 space-y-2">
                     <div className="flex justify-between items-center text-xs">
                       <span className="text-gray-400">
                         <span className={trainProgress.split === "train" ? "text-brand-400" : "text-emerald-400"}>
                           {trainProgress.split}
                         </span>
                         {" "} {t("training.batch_progress")}
                       </span>
                       <span className="tabular-nums text-gray-300">
                         {trainProgress.batch_cur} / {trainProgress.batch_total}
                       </span>
                     </div>

                    <ProgressBar
                      pct={trainProgress.pct}
                      color={trainProgress.split === "train" ? "bg-brand-400" : "bg-emerald-400"}
                    />
                  </div>

                  {/* 속도 / ETA */}
                  <div className="flex gap-4 text-xs">
                     <div>
                       <span className="text-gray-500">{t("training.speed")} </span>
                       <span className="text-gray-200 tabular-nums font-mono">
                         {trainProgress.speed_it_s > 0
                           ? `${trainProgress.speed_it_s.toFixed(2)} it/s`
                           : "—"}
                       </span>
                     </div>
                     <div>
                       <span className="text-gray-500">{t("training.eta")} </span>
                       <span className="text-gray-200 tabular-nums font-mono">
                         {fmtEta(trainProgress.eta_sec)}
                       </span>
                     </div>

                  </div>
                </>
              )}
            </div>
          )}

          {/* 차트 */}
          <div className="card">
            <MetricsChart
              metrics={trainMetrics}
              phase1Epochs={phase1Epochs}
              progress={trainProgress}
              running={running}
            />
          </div>
        </div>
      </div>

       {/* ── 얼굴 크롭 전처리 ── */}
       <div className="card space-y-3">
         <button
           className="flex items-center justify-between w-full text-left"
           onClick={() => setSegLogOpen((v) => !v)}
         >
           <div className="flex items-center gap-2">
             <p className="text-sm font-medium text-gray-200">{t("training.seg_title")}</p>
             <span className={`text-[10px] px-2 py-0.5 rounded-full ${
               segState === "running" ? "bg-brand-900 text-brand-300" :
               segState === "done"    ? "bg-green-900 text-green-300" :
               segState === "failed"  ? "bg-red-900 text-red-300"     :
               "bg-gray-800 text-gray-400"
             }`}>
               {t(`training.seg_status_${segState}`)}
             </span>
           </div>
           <span className="text-gray-500 text-xs">{segLogOpen ? "▲" : "▼"}</span>
         </button>

         {segLogOpen && (
           <div className="space-y-3 pt-1">
             <p className="text-[11px] text-gray-500 leading-4">{t("training.seg_hint")}</p>

             <div className="grid grid-cols-2 gap-2">
               <div>
                 <label className="label-text">{t("training.seg_input_dir")}</label>
                 <input value={segInputDir}
                   onChange={(e) => setSegInputDir(e.target.value)}
                   className="input font-mono text-xs"
                   disabled={segState === "running"}
                 />
               </div>
               <div>
                 <label className="label-text">{t("training.seg_output_dir")}</label>
                 <input value={segOutputDir}
                   onChange={(e) => setSegOutputDir(e.target.value)}
                   className="input font-mono text-xs"
                   disabled={segState === "running"}
                 />
               </div>
             </div>

             <div>
               <label className="label-text">{t("training.seg_backend")}</label>
               <select value={segBackend}
                 onChange={(e) => setSegBackend(e.target.value as "cascade"|"yolo")}
                 className="input"
                 disabled={segState === "running"}
               >
                 <option value="cascade">{t("training.seg_backend_cascade")}</option>
                 <option value="yolo">{t("training.seg_backend_yolo")}</option>
               </select>
             </div>

             {/* 진행률 */}
             {(segState === "running" || (segState === "done" && segPct > 0)) && (
               <div className="space-y-1.5">
                 <div className="flex justify-between text-xs text-gray-400">
                   <span>{segClass || "처리 중..."}</span>
                   <span className="tabular-nums">
                     {segPct.toFixed(1)}%
                     {segEta > 0 ? ` · ${fmtEta(segEta)}` : ""}
                   </span>
                 </div>
                 <ProgressBar pct={segPct} color="bg-teal-500" />
               </div>
             )}

             <div className="flex gap-2">
               {segState !== "running" ? (
                 <button onClick={startSeg} className="btn-primary flex-1">
                   {t("training.seg_start_btn")}
                 </button>
               ) : (
                 <button onClick={stopSeg} className="btn-danger flex-1">
                   {t("training.seg_stop_btn")}
                 </button>
               )}
             </div>

             <JobConsole
               title={t("training.seg_log_title")}
               jobPath="/segmentation/logs"
               onState={(s) => {
                 setSegState(s as "idle"|"running"|"done"|"failed");
               }}
               onMessage={(msg) => {
                 if (msg.type === "seg_progress") {
                   const d = msg.data as { pct?: number; class?: string; eta_sec?: number };
                   setSegPct(d.pct ?? 0);
                   setSegClass(d.class ?? "");
                   setSegEta(d.eta_sec ?? -1);
                 }
               }}
             />
           </div>
         )}

         {/* 크롭 데이터 학습 토글 */}
         {(segState === "done") && (
           <label className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer pt-1 border-t border-gray-800">
             <input type="checkbox" checked={useFaceCrop}
               onChange={(e) => setUseFaceCrop(e.target.checked)}
               className="accent-teal-500" disabled={running} />
             <span>{t("training.seg_use_cropped")}</span>
             <span className="text-gray-600 font-mono text-[10px]">{segOutputDir}</span>
           </label>
         )}
       </div>

       {/* ── 학습 로그 ── */}
       <div className="card">
         <JobConsole
           title={t("training.log_title")}
           jobPath="/training/logs"
           onState={(s) => {
             setTrainState(s);
             if (s !== "running") setTrainProgress(null);
           }}
           onMetric={(m) => pushMetric(m)}
           onProgress={(p) => setTrainProgress(p)}
         />
       </div>

    </div>
  );
}
