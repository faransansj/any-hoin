/**
 * MetricsChart — 학습 loss/acc 실시간 라인차트 (Recharts)
 * Phase 1/2 구분선 포함
 */
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainMetric, TrainProgress } from "../types";

interface Props {
  metrics:       TrainMetric[];
  phase1Epochs?: number;
  progress?:     TrainProgress | null;
  running?:      boolean;
}

const fmt2 = (v: number) => v.toFixed(4);

function LiveProgressChart({ progress }: { progress: TrainProgress | null }) {
  if (!progress) {
    return (
      <div className="flex items-center justify-center h-52 text-gray-600 text-sm">
        첫 batch 진행률을 기다리는 중입니다
      </div>
    );
  }

  const color = progress.split === "train" ? "#8bafff" : "#34d399";
  const pct = Math.min(100, Math.max(0, progress.pct));

  return (
    <div className="h-52 flex flex-col justify-center gap-5">
      <div>
        <div className="flex items-center justify-between text-xs mb-2">
          <span className={progress.split === "train" ? "text-brand-300" : "text-emerald-300"}>
            live {progress.split} progress
          </span>
          <span className="text-gray-500 tabular-nums">
            {progress.batch_cur} / {progress.batch_total} batches
          </span>
        </div>
        <div className="h-3 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
        <div className="mt-1 text-right text-xs text-gray-500 tabular-nums">{pct}%</div>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-3">
          <p className="text-[10px] text-gray-500 mb-1">running loss</p>
          <p className="text-sm font-semibold text-gray-100 tabular-nums">
            {progress.avg_loss !== undefined ? progress.avg_loss.toFixed(4) : "-"}
          </p>
        </div>
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-3">
          <p className="text-[10px] text-gray-500 mb-1">running acc</p>
          <p className="text-sm font-semibold text-gray-100 tabular-nums">
            {progress.avg_acc !== undefined ? `${(progress.avg_acc * 100).toFixed(2)}%` : "-"}
          </p>
        </div>
        <div className="rounded-lg bg-gray-900 border border-gray-800 p-3">
          <p className="text-[10px] text-gray-500 mb-1">speed</p>
          <p className="text-sm font-semibold text-gray-100 tabular-nums">
            {progress.speed_it_s > 0 ? `${progress.speed_it_s.toFixed(2)} it/s` : "-"}
          </p>
        </div>
      </div>

      <p className="text-[11px] text-gray-500">
        Loss/Accuracy 라인 차트는 epoch가 끝나 metric이 확정되면 자동으로 표시됩니다.
      </p>
    </div>
  );
}

export default function MetricsChart({ metrics, phase1Epochs, progress, running }: Props) {
  if (metrics.length === 0) {
    if (running) return <LiveProgressChart progress={progress ?? null} />;
    return (
      <div className="flex items-center justify-center h-52 text-gray-600 text-sm">
        학습 시작 후 첫 epoch가 끝나면 차트가 표시됩니다
      </div>
    );
  }

  // x축: 누적 epoch 번호
  const data = metrics.map((m, i) => ({
    x:          i + 1,
    train_loss: m.train_loss,
    val_loss:   m.val_loss,
    train_acc:  m.train_acc,
    val_acc:    m.val_acc,
    phase:      m.phase,
  }));

  // Phase 2 시작 지점
  const phase2Start = phase1Epochs
    ? phase1Epochs + 0.5
    : data.findIndex((d) => d.phase === 2) + 0.5;

  return (
    <div className="space-y-4">
      {/* Loss */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Loss</p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} tickFormatter={fmt2} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={fmt2}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {phase2Start > 0 && (
              <ReferenceLine x={phase2Start} stroke="#6d8fff" strokeDasharray="4 4" label={{ value: "Phase 2", fill: "#6d8fff", fontSize: 10 }} />
            )}
            <Line type="monotone" dataKey="train_loss" stroke="#f59e0b" dot={false} name="train" strokeWidth={1.5} />
            <Line type="monotone" dataKey="val_loss"   stroke="#ef4444" dot={false} name="val"   strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy */}
      <div>
        <p className="text-xs text-gray-400 mb-1">Accuracy</p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="x" tick={{ fontSize: 10, fill: "#6b7280" }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#6b7280" }} tickFormatter={fmt2} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", fontSize: 11 }}
              formatter={fmt2}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {phase2Start > 0 && (
              <ReferenceLine x={phase2Start} stroke="#6d8fff" strokeDasharray="4 4" />
            )}
            <Line type="monotone" dataKey="train_acc" stroke="#34d399" dot={false} name="train" strokeWidth={1.5} />
            <Line type="monotone" dataKey="val_acc"   stroke="#60a5fa" dot={false} name="val"   strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
