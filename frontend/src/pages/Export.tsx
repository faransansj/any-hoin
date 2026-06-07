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
import { useTranslation } from "react-i18next";
import type { ExportStatus, ModelMap, ModelsResponse, QuantFormat } from "../types";

const QUANT_OPTIONS: {
  value: QuantFormat;
  label: string;
  recommended?: boolean;
  color: string;
}[] = [
  {
    value: "fp16",
    label: "FP16",
    recommended: true,
    color: "text-yellow-300",
  },
  {
    value: "int8",
    label: "INT8",
    color: "text-orange-300",
  },
  {
    value: "int4",
    label: "INT4",
    color: "text-red-300",
  },
  {
    value: "int2",
    label: "INT2",
    color: "text-pink-400",
  },
];

const QUANT_KEYS: QuantFormat[] = ["fp16", "int8", "int4", "int2"];

export default function Export() {
  const { t, i18n } = useTranslation();
  const isKo = i18n.resolvedLanguage?.startsWith("ko") ?? false;
  const tx = (ko: string, en: string) => (isKo ? ko : en);
  const { quantState, onnxState, hoinState, setQuantState, setOnnxState, setHoinState } = useJobStore();
  const [models,    setModels]    = useState<ModelMap | null>(null);
  const [configAcc, setConfigAcc] = useState<number | null>(null);
  const [format,    setFormat]    = useState<QuantFormat>("fp16");
  const [opset,     setOpset]     = useState(18);
  const [logTab,    setLogTab]    = useState<"quant" | "onnx" | "hoin">("quant");

  async function loadModels() {
    const r = await api.get<ModelsResponse>("/export/models");
    setModels(r.models);
    setConfigAcc(r.config_acc ?? null);
  }

  useEffect(() => {
    loadModels();
    api.get<ExportStatus>("/export/status")
      .then((r) => { setQuantState(r.quant.state); setOnnxState(r.onnx.state); setHoinState(r.hoin.state); })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (quantState === "done" || onnxState === "done" || hoinState === "done") loadModels();
  }, [quantState, onnxState, hoinState]);

  const selectedOpt = QUANT_OPTIONS.find((o) => o.value === format)!;
  const quantMeta: Record<QuantFormat, { desc: string; useCase: string; risk: string }> = {
    fp16: {
      desc: tx("약 2× 압축", "About 2× smaller"),
      useCase: tx("GPU/ONNX 배포 전 기본 권장 옵션. 정확도 손실이 거의 없습니다.", "Default recommended option before GPU/ONNX deployment with minimal accuracy loss."),
      risk: tx("낮음", "Low"),
    },
    int8: {
      desc: tx("약 4× 압축", "About 4× smaller"),
      useCase: tx("모바일/CPU 추론처럼 용량과 속도가 중요할 때 실험용으로 적합합니다.", "Good for experiments where size and speed matter, such as mobile/CPU inference."),
      risk: tx("중간", "Medium"),
    },
    int4: {
      desc: tx("약 8× 압축", "About 8× smaller"),
      useCase: tx("모델 크기를 크게 줄여야 할 때만 사용합니다. 결과 검증이 필요합니다.", "Use only when model size must be reduced significantly; result validation is required."),
      risk: tx("높음", "High"),
    },
    int2: {
      desc: tx("약 16× 압축", "About 16× smaller"),
      useCase: tx("극단적인 용량 제한 실험용입니다. 실제 서비스 기본값으로는 권장하지 않습니다.", "For extreme size-constrained experiments; not recommended as a production default."),
      risk: tx("매우 높음", "Very high"),
    },
  };
  const quantRunning = quantState === "running";
  const onnxRunning  = onnxState  === "running";
  const hoinRunning  = hoinState  === "running";

  // 비교 테이블: fp32 + 존재하는 양자화 + onnx
  const tableRows = (["fp32", ...QUANT_KEYS, "onnx"] as const).filter(
    (k) => models?.[k]?.exists
  );
  const packageReady = Boolean(
    models &&
    (models.fp32.exists || models.onnx.exists || QUANT_KEYS.some((key) => models[key]?.exists)) &&
    models.class_map.exists &&
    models.config.exists
  );

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-bold text-white">{t("export.title")}</h1>
        <p className="text-sm text-gray-400 mt-0.5">{t("export.subtitle")}</p>
      </div>

      {models && (
        <div className="card">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-semibold text-gray-200">{t("export.package_title")}</p>
              <p className="mt-0.5 text-xs leading-5 text-gray-500">{t("export.package_desc")}</p>
              {!packageReady && (
                <p className="mt-1 text-xs text-amber-400">{t("export.package_missing")}</p>
              )}
            </div>
            {packageReady ? (
              <a
                href="/api/export/download-package"
                download
                className="btn-primary shrink-0 text-center text-sm"
              >
                {t("export.package_download")}
              </a>
            ) : (
              <button type="button" disabled className="btn-ghost shrink-0 text-sm">
                {t("export.package_download")}
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── 카드 행 ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* ── 양자화 카드 ── */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-white">{t("export.quant_title")}</span>
            <StatusBadge state={quantState} />
          </div>

          {/* 드롭다운 */}
          <div>
            <label className="label-text mb-1 block">{t("export.format_label")}</label>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as QuantFormat)}
              disabled={quantRunning}
              className="input w-full text-sm"
            >
              {QUANT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label} — {quantMeta[o.value].desc}
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
                        <span className="rounded-full bg-green-500/15 px-1.5 py-0.5 text-[10px] text-green-300">{tx("권장", "Recommended")}</span>
                      )}
                    </div>
                    <span>
                      {entry?.exists
                        ? <span className="text-green-400">{entry.size_mb} MB ✓</span>
                        : <span className="text-gray-600">{t("export.not_exists")}</span>}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-gray-400">{quantMeta[o.value].useCase}</p>
                  <p className="mt-1 text-[10px] text-gray-500">{tx("정확도 손실 위험", "Accuracy risk")}: {quantMeta[o.value].risk} · {quantMeta[o.value].desc}</p>
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
              {tx(`${selectedOpt.label} 변환 시작`, `Start ${selectedOpt.label} Conversion`)}
            </button>
          ) : (
            <button onClick={() => api.post("/export/quant/stop")} className="btn-danger w-full text-sm">
              {t("export.quant_stop_btn")}
            </button>
          )}

          {/* 현재 선택 형식 다운로드 */}
          {models?.[format]?.exists && (
            <a
              href={`/api/export/download/${models[format]!.filename}`}
              download
              className="block text-center btn-ghost text-xs py-1"
            >
              {t("export.download_btn", { format: selectedOpt.label })}
            </a>
          )}
        </div>

        {/* ── ONNX 카드 ── */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-blue-300">{t("export.onnx_title")}</span>
            <StatusBadge state={onnxState} />
          </div>

          <div className="space-y-1 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>{t("export.status")}</span>
              <span className={models?.onnx.exists ? "text-green-400" : "text-gray-600"}>
                {models?.onnx.exists ? t("export.exists") : t("export.not_exists")}
              </span>
            </div>
            <div className="flex justify-between">
              <span>{t("export.size")}</span>
              <span>{models?.onnx.size_mb != null ? `${models.onnx.size_mb} MB` : "—"}</span>
            </div>
          </div>

          <div>
            <label className="label-text mb-1 block">{t("export.opset_label")}</label>
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
              {t("export.onnx_start_btn")}
            </button>
          ) : (
            <button onClick={() => api.post("/export/onnx/stop")} className="btn-danger w-full text-sm">
              {t("export.onnx_stop_btn")}
            </button>
          )}

          {models?.onnx.exists && (
            <div className="space-y-2">
              <a
                href={`/api/export/download/${models.onnx.filename}`}
                download
                className="block text-center btn-ghost text-xs py-1"
              >
                {t("export.onnx_download")}
              </a>
              {models.onnx_data?.exists && (
                <a
                  href={`/api/export/download/${models.onnx_data.filename}`}
                  download
                  className="block text-center btn-ghost text-xs py-1"
                >
                  {tx("ONNX data 다운로드", "Download ONNX data")}
                </a>
              )}
            </div>
          )}
        </div>

        {/* ── hoin 카드 ── */}
        <div className="card space-y-4">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sm text-green-300">hoin 서비스 패키지</span>
            <StatusBadge state={hoinState} />
          </div>

          <div className="rounded-lg bg-green-950/20 border border-green-900/50 px-3 py-2 text-xs text-gray-300 space-y-2">
            <p className="leading-5">
              hoin에서 바로 쓰는 모델 폴더/zip을 생성합니다. ONNX, 외부 weights,
              <code className="mx-1 text-green-200">hoin-model.json</code>,
              <code className="mx-1 text-green-200">class_map.json</code>을 함께 내보냅니다.
            </p>
            <div className="flex justify-between">
              <span>패키지</span>
              <span className={models?.hoin.exists ? "text-green-400" : "text-gray-600"}>
                {models?.hoin.exists ? `✓ ${models.hoin.size_mb} MB` : "미생성"}
              </span>
            </div>
            <div className="text-[11px] text-gray-500">
              출력: <code>models/any-hoin/</code> 및 <code>models/any-hoin-hoin-model.zip</code>
            </div>
          </div>

          {!hoinRunning ? (
            <button
              onClick={() => { setLogTab("hoin"); api.post("/export/hoin", { opset, model_name: "any-hoin" }); }}
              disabled={!models?.fp32.exists}
              className="btn-primary w-full text-sm"
            >
              hoin용으로 한 번에 내보내기
            </button>
          ) : (
            <button onClick={() => api.post("/export/hoin/stop")} className="btn-danger w-full text-sm">
              중단
            </button>
          )}

          {models?.hoin.exists && (
            <a
              href="/api/export/hoin/download"
              download
              className="block text-center btn-ghost text-xs py-1"
            >
              ⬇ hoin 패키지 ZIP 다운로드
            </a>
          )}
        </div>
      </div>

      {models && (
        <div className="card">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-gray-200">{t("export.metadata_title")}</p>
              <p className="mt-0.5 text-xs text-gray-500">{t("export.metadata_desc")}</p>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3">
            {([
              ["class_map", "class_map.json", tx("추론 라벨 매핑", "Inference label map")],
              ["config", "config.json", tx("학습/백본 설정", "Training/backbone config")],
              ["onnx_data", "best_model.onnx.data", tx("ONNX 외부 데이터", "ONNX external data")],
            ] as const).map(([key, label, desc]) => {
              const entry = models[key];
              return (
                <div key={key} className="rounded-lg bg-gray-950 border border-gray-800 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="font-semibold text-xs text-gray-200 truncate">{label}</p>
                      <p className="mt-1 text-[11px] text-gray-500">{desc}</p>
                      <p className={`mt-2 text-[11px] ${entry?.exists ? "text-green-400" : "text-gray-600"}`}>
                        {entry?.exists ? `${entry.size_mb} MB` : t("export.not_exists")}
                      </p>
                    </div>
                    {entry?.exists && (
                      <a
                        href={`/api/export/download/${entry.filename}`}
                        download
                        className="btn-ghost shrink-0 text-xs py-1 px-2"
                      >
                        {t("export.metadata_download")}
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── 비교 테이블 ── */}
      {models && tableRows.length > 1 && (
        <div className="card">
          <p className="text-sm font-medium text-gray-200 mb-3">{t("export.compare_title")}</p>
          <table className="w-full text-xs text-gray-300">
            <thead>
              <tr className="text-gray-500 border-b border-gray-700">
                <th className="text-left pb-2">{t("export.table_format")}</th>
                <th className="text-right pb-2">{t("export.table_size")}</th>
                <th className="text-right pb-2">{t("export.table_ratio")}</th>
                <th className="text-right pb-2">{t("export.table_acc")}</th>
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
            <p className="mt-2 text-xs text-gray-600">{t("export.acc_hint")}</p>
          )}
        </div>
      )}

      {/* ── 로그 ── */}
      <div className="card">
        <div className="flex gap-2 mb-2">
          {(["quant", "onnx", "hoin"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setLogTab(tab)}
              className={`text-xs px-3 py-1 rounded-md transition-colors ${
                logTab === tab ? "bg-brand-600 text-white" : "bg-gray-800 text-gray-400"
              }`}
            >
              {tab === "quant" ? t("export.log_tab_quant") : tab === "onnx" ? t("export.log_tab_onnx") : "hoin 로그"}
            </button>
          ))}
        </div>
        <JobConsole
          key={logTab}
          jobPath={`/export/logs/${logTab}`}
          onState={(s) => {
            if (logTab === "quant") setQuantState(s);
            else if (logTab === "onnx") setOnnxState(s);
            else setHoinState(s);
          }}
        />
      </div>
    </div>
  );
}
