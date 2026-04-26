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
import type { JobState, TrainMetric, TrainProgress } from "../types";

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

// ── 컴포넌트 ─────────────────────────────────────────────

export default function Training() {
  const { t } = useTranslation();
  const {
    trainState, setTrainState,
    trainMetrics, bestValAcc, pushMetric, resetMetrics,
    trainProgress, setTrainProgress,
  } = useJobStore();

  const [batchSize,     setBatchSize]     = useState(32);
  const [phase1Epochs,  setPhase1Epochs]  = useState(5);
  const [phase2Epochs,  setPhase2Epochs]  = useState(30);
  const [phase2Lr,      setPhase2Lr]      = useState("1e-5");
  const [patience,      setPatience]      = useState(7);
  const [device,        setDevice]        = useState("auto");
  const [finetune,      setFinetune]      = useState(false);
  const [noAmp,         setNoAmp]         = useState(false);

  useEffect(() => {
    api.get<{ state: JobState; epoch_count: number; best_val_acc: number }>("/training/status")
      .then((r) => setTrainState(r.state))
      .catch(console.error);

    api.get<{ metrics: TrainMetric[] }>("/training/metrics")
      .then((r) => r.metrics.forEach(pushMetric))
      .catch(console.error);
  }, []);

  async function startTraining() {
    resetMetrics();
    await api.post("/training/start", {
      batch_size:    batchSize,
      phase1_epochs: phase1Epochs,
      phase2_epochs: phase2Epochs,
      phase2_lr:     parseFloat(phase2Lr),
      patience,
      device,
      finetune,
      no_amp: noAmp,
    });
  }

  async function stopTraining() {
    await api.post("/training/stop");
  }

  const running = trainState === "running";
  const latestMetric = trainMetrics.at(-1);

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
             <label className="label-text">{t("training.device")}</label>
             <select value={device} onChange={(e) => setDevice(e.target.value)}
               className="input" disabled={running}>

              <option value="auto">auto</option>
              <option value="cuda">CUDA</option>
              <option value="mps">MPS (Apple)</option>
              <option value="xpu">XPU (Intel Arc)</option>
              <option value="cpu">CPU</option>
            </select>
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


          <div className="space-y-1 pt-1">
             <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
               <input type="checkbox" checked={finetune} onChange={(e) => setFinetune(e.target.checked)}
                 className="accent-brand-500" disabled={running} />
               {t("training.finetune")}
             </label>
             <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
               <input type="checkbox" checked={noAmp} onChange={(e) => setNoAmp(e.target.checked)}
                 className="accent-brand-500" disabled={running} />
               {t("training.no_amp")}
             </label>

          </div>

           <div className="pt-1">
             {!running ? (
               <button onClick={startTraining} className="btn-primary w-full">{t("training.start_btn")}</button>
             ) : (
               <button onClick={stopTraining} className="btn-danger w-full">{t("training.stop_btn")}</button>
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
            <MetricsChart metrics={trainMetrics} phase1Epochs={phase1Epochs} />
          </div>
        </div>
      </div>

       {/* ── 로그 ── */}
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
