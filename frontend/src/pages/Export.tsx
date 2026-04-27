/**
 * Export 페이지
 * - 양자화 (FP16 / INT8 / INT4 / INT2) — 드롭다운 선택
 * - ONNX 변환 — 별도 카드
 * - 비교 테이블 + 다운로드
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import JobConsole from "../components/JobConsole";
import StatusBadge from "../components/StatusBadge";
import { useJobStore } from "../store/jobStore";
import type { ExportStatus, ModelMap, ModelsResponse, QuantFormat } from "../types";

const QUANT_OPTIONS: {
  value: QuantFormat;
  label: string;
  desc: string;
  useCase: string;
  risk: string;
  recommended?: boolean;
  color: string;
}[] = [
  {
    value: "fp16",
    label: "FP16",
    desc: "약 2× 압축",
    useCase: "GPU/ONNX 배포 전 기본 권장 옵션. 정확도 손실이 거의 없습니다.",
    risk: "낮음",
    recommended: true,
    color: "text-yellow-300",
  },
  {
    value: "int8",
    label: "INT8",
    desc: "약 4× 압축",
    useCase: "모바일/CPU 추론처럼 용량과 속도가 중요할 때 실험용으로 적합합니다.",
    risk: "중간",
    color: "text-orange-300",
  },
  {
    value: "int4",
    label: "INT4",
    desc: "약 8× 압축",
    useCase: "모델 크기를 크게 줄여야 할 때만 사용합니다. 결과 검증이 필요합니다.",
    risk: "높음",
    color: "text-red-300",
  },
  {
    value: "int2",
    label: "INT2",
    desc: "약 16× 압축",
    useCase: "극단적인 용량 제한 실험용입니다. 실제 서비스 기본값으로는 권장하지 않습니다.",
    risk: "매우 높음",
    color: "text-pink-400",
  },
];

const QUANT_KEYS: QuantFormat[] = ["fp16", "int8", "int4", "int2"];

export default function Export() {
  const { quantState, onnxState, setQuantState, setOnnxState } = useJobStore();
  const [models,    setModels]    = useState<ModelMap | null>(null);
  const [configAcc, setConfigAcc] = useState<number | null>(null);
  const [format,    setFormat]    = useState<QuantFormat>("fp16");
  const [opset,     setOpset]     = useState(18);
  const [logTab,    setLogTab]    = useState<"quant" | "onnx">("quant");

  async function loadModels() {
    const r = await api.get<ModelsResponse>("/export/models");
    setModels(r.models);
    setConfigAcc(r.config_acc ?? null);
  }

  useEffect(() => {
    loadModels();
    api.get<ExportStatus>("/export/status")
      .then((r) => { setQuantState(r.quant.state); setOnnxState(r.onnx.state); })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (quantState === "done" || onnxState === "done") loadModels();
  }, [quantState, onnxState]);

  const selectedOpt = QUANT_OPTIONS.find((o) => o.value === format)!;
  const quantRunning = quantState === "running";
  const onnxRunning  = onnxState  === "running";

  // 비교 테이블: fp32 + 존재하는 양자화 + onnx
  const tableRows = (["fp32", ...QUANT_KEYS, "onnx"] as const).filter(
    (k) => models?.[k]?.exists
  );

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-bold text-white">Export</h1>
        <p className="text-sm text-gray-400 mt-0.5">모델 양자화 및 배포 형식 변환</p>
      </div>

      {/* ── 카드 행 ── */}
      <div className="grid grid-cols-2 gap-4">

        {/* ── 양자화 카드 ── */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-white">양자화</span>
            <StatusBadge state={quantState} />
          </div>

          {/* 드롭다운 */}
          <div>
            <label className="label-text mb-1 block">형식 선택</label>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as QuantFormat)}
              disabled={quantRunning}
              className="input w-full text-sm"
            >
              {QUANT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label} — {o.desc}
                </option>
              ))}
            </select>
          </div>

          {/* 선택된 형식 정보 */}
          <div className="rounded-lg bg-gray-800/60 px-3 py-2 space-y-2 text-xs text-gray-400">
            {QUANT_OPTIONS.map((o) => {
              const entry = models?.[o.value];
              return (
                <div key={o.value} className={`rounded-md border p-2 ${o.value === format ? "border-brand-500/60 bg-brand-600/10" : "border-gray-700/70 bg-gray-900/40"}`}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${o.color}`}>{o.label}</span>
                      {o.recommended && (
                        <span className="rounded-full bg-green-500/15 px-1.5 py-0.5 text-[10px] text-green-300">권장</span>
                      )}
                    </div>
                    <span>
                      {entry?.exists
                        ? <span className="text-green-400">{entry.size_mb} MB ✓</span>
                        : <span className="text-gray-600">미변환</span>}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-gray-400">{o.useCase}</p>
                  <p className="mt-1 text-[10px] text-gray-500">정확도 손실 위험: {o.risk} · {o.desc}</p>
                </div>
              );
            })}
          </div>

          {/* 변환 버튼 */}
          {!quantRunning ? (
            <button
              onClick={() => api.post("/export/quant", { format })}
              disabled={!models?.fp32.exists}
              className="btn-primary w-full text-sm"
            >
              {selectedOpt.label} 변환 시작
            </button>
          ) : (
            <button onClick={() => api.post("/export/quant/stop")} className="btn-danger w-full text-sm">
              중단
            </button>
          )}

          {/* 현재 선택 형식 다운로드 */}
          {models?.[format]?.exists && (
            <a
              href={`/api/export/download/${models[format]!.filename}`}
              download
              className="block text-center btn-ghost text-xs py-1"
            >
              ⬇ {selectedOpt.label} 다운로드
            </a>
          )}
        </div>

        {/* ── ONNX 카드 ── */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-blue-300">ONNX (배포)</span>
            <StatusBadge state={onnxState} />
          </div>

          <div className="space-y-1 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>상태</span>
              <span className={models?.onnx.exists ? "text-green-400" : "text-gray-600"}>
                {models?.onnx.exists ? "✓ 존재" : "없음"}
              </span>
            </div>
            <div className="flex justify-between">
              <span>크기</span>
              <span>{models?.onnx.size_mb != null ? `${models.onnx.size_mb} MB` : "—"}</span>
            </div>
          </div>

          <div>
            <label className="label-text mb-1 block">Opset</label>
            <input
              type="number" value={opset} min={11} max={20}
              onChange={(e) => setOpset(+e.target.value)}
              className="input w-full text-sm"
              disabled={onnxRunning}
            />
          </div>

          {!onnxRunning ? (
            <button
              onClick={() => api.post("/export/onnx", { opset })}
              disabled={!models?.fp32.exists}
              className="btn-primary w-full text-sm"
            >
              ONNX 변환 시작
            </button>
          ) : (
            <button onClick={() => api.post("/export/onnx/stop")} className="btn-danger w-full text-sm">
              중단
            </button>
          )}

          {models?.onnx.exists && (
            <a
              href={`/api/export/download/${models.onnx.filename}`}
              download
              className="block text-center btn-ghost text-xs py-1"
            >
              ⬇ ONNX 다운로드
            </a>
          )}
        </div>
      </div>

      {/* ── 비교 테이블 ── */}
      {models && tableRows.length > 1 && (
        <div className="card">
          <p className="text-sm font-medium text-gray-200 mb-3">모델 비교</p>
          <table className="w-full text-xs text-gray-300">
            <thead>
              <tr className="text-gray-500 border-b border-gray-700">
                <th className="text-left pb-2">형식</th>
                <th className="text-right pb-2">크기</th>
                <th className="text-right pb-2">압축률</th>
                <th className="text-right pb-2">정확도</th>
                <th className="text-right pb-2"></th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((k) => {
                const e = models[k];
                if (!e.exists) return null;
                const ratio = models.fp32.size_mb && e.size_mb
                  ? Math.round((e.size_mb / models.fp32.size_mb) * 100)
                  : 100;
                const opt = QUANT_OPTIONS.find((o) => o.value === k);
                const colorCls = opt?.color ?? "text-gray-300";

                let accLabel: React.ReactNode = "—";
                if (k === "fp32" && configAcc != null) {
                  accLabel = <span className="text-green-400">{(configAcc * 100).toFixed(2)}%</span>;
                } else if (k === "fp16" && configAcc != null) {
                  accLabel = <span className="text-yellow-300">≈ {(configAcc * 100).toFixed(2)}%</span>;
                } else if (k === "onnx" && configAcc != null) {
                  accLabel = <span className="text-blue-300">≈ {(configAcc * 100).toFixed(2)}%</span>;
                }

                return (
                  <tr key={k} className="border-b border-gray-800">
                    <td className={`py-1.5 font-medium ${colorCls}`}>{k.toUpperCase()}</td>
                    <td className="text-right">{e.size_mb} MB</td>
                    <td className="text-right text-gray-500">{ratio}%</td>
                    <td className="text-right">{accLabel}</td>
                    <td className="text-right">
                      <a
                        href={`/api/export/download/${e.filename}`}
                        download
                        className="text-gray-500 hover:text-gray-300 transition-colors"
                      >
                        ⬇
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {configAcc == null && (
            <p className="mt-2 text-xs text-gray-600">정확도: 학습 완료 후 config.json에서 로드됩니다</p>
          )}
        </div>
      )}

      {/* ── 로그 ── */}
      <div className="card">
        <div className="flex gap-2 mb-2">
          {(["quant", "onnx"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setLogTab(t)}
              className={`text-xs px-3 py-1 rounded-md transition-colors ${
                logTab === t ? "bg-brand-600 text-white" : "bg-gray-800 text-gray-400"
              }`}
            >
              {t === "quant" ? "양자화" : "ONNX"} 로그
            </button>
          ))}
        </div>
        <JobConsole
          key={logTab}
          jobPath={`/export/logs/${logTab}`}
          onState={(s) => logTab === "quant" ? setQuantState(s) : setOnnxState(s)}
        />
      </div>
    </div>
  );
}
