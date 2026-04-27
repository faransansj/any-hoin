/**
 * Inference 페이지
 * - 이미지 드롭존
 * - Top-5 예측 바 차트
 */
import { useCallback, useEffect, useState } from "react";
import { useDropzone } from "react-dropzone";
import { api } from "../api";
import { useTranslation } from "react-i18next";
import type { InferenceModelInfo } from "../types";

interface Prediction {
  rank:         number;
  character:    string;
  display_name?: string;
  confidence:   number;
}

interface PredictResult {
  filename: string;
  top_k:    Prediction[];
}

function fmtPct(value: number | null | undefined) {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
}

export default function Inference() {
  const { t } = useTranslation();
  const [result,   setResult]   = useState<PredictResult | null>(null);
  const [preview,  setPreview]  = useState<string | null>(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<InferenceModelInfo | null>(null);

  useEffect(() => {
    api.get<InferenceModelInfo>("/inference/model-info")
      .then(setModelInfo)
      .catch(console.error);
  }, []);

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);
      const r = await api.upload<PredictResult>("/inference/predict", form);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles: 1,
  });

  const top1 = result?.top_k[0];
  const apiExample: PredictResult = result ?? {
    filename: "sample.webp",
    top_k: [
      {
        rank: 1,
        character: "hina_(blue_archive)",
        display_name: "Hina (Blue Archive)",
        confidence: 0.9274,
      },
      {
        rank: 2,
        character: "yuuka_(blue_archive)",
        display_name: "Yuuka (Blue Archive)",
        confidence: 0.0418,
      },
    ],
  };
  const apiExampleJson = JSON.stringify(apiExample, null, 2);

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-white">{t("inference.title")}</h1>
        <p className="text-sm text-gray-400 mt-0.5">{t("inference.subtitle")}</p>
      </div>

      <div className="card">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-gray-200">{t("inference.model_info_title")}</p>
            <p className="text-xs text-gray-500 mt-0.5">{t("inference.model_info_desc")}</p>
          </div>
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            modelInfo?.model_ready ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
          }`}>
            {modelInfo?.model_ready ? t("inference.model_ready") : t("inference.model_missing")}
          </span>
        </div>
        <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
            <p className="text-gray-500">Backend</p>
            <p className="mt-1 font-semibold text-gray-100">{modelInfo?.loaded_backend ?? modelInfo?.preferred_backend ?? "—"}</p>
          </div>
          <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
            <p className="text-gray-500">Classes</p>
            <p className="mt-1 font-semibold text-gray-100 tabular-nums">{modelInfo?.num_classes ?? "—"}</p>
          </div>
          <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
            <p className="text-gray-500">Best Val</p>
            <p className="mt-1 font-semibold text-gray-100 tabular-nums">{fmtPct(modelInfo?.best_val_acc)}</p>
          </div>
          <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
            <p className="text-gray-500">Test Acc</p>
            <p className="mt-1 font-semibold text-gray-100 tabular-nums">{fmtPct(modelInfo?.test_acc)}</p>
          </div>
          <div className="rounded-lg bg-gray-950 border border-gray-800 p-3">
            <p className="text-gray-500">Files</p>
            <p className="mt-1 font-semibold text-gray-100">
              {[
                modelInfo?.fp32_available ? "FP32" : null,
                modelInfo?.fp16_available ? "FP16" : null,
                modelInfo?.onnx_available ? "ONNX" : null,
              ].filter(Boolean).join(" / ") || "—"}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ── 왼쪽: 드롭존 및 프리뷰 ── */}
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors min-h-[400px] flex flex-col items-center justify-center ${
              isDragActive
                ? "border-brand-500 bg-brand-600/10"
                : "border-gray-700 hover:border-gray-500 bg-gray-900"
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img src={preview} alt="preview" className="max-h-full max-w-full rounded-lg object-contain" />
            ) : (
              <div className="space-y-2">
                <div className="text-4xl">🖼️</div>
                <p className="text-sm text-gray-300">
                  {isDragActive ? t("inference.drop_hint") : t("inference.drop_click")}
                </p>
                <p className="text-xs text-gray-600">{t("inference.format_hint")}</p>
              </div>
            )}
          </div>
          
          {preview && (
            <button
              onClick={() => { setResult(null); setPreview(null); }}
              className="btn-ghost text-xs w-full"
            >
              {t("inference.reset_btn")}
            </button>
          )}
        </div>

        {/* ── 오른쪽: 결과 영역 ── */}
        <div className="space-y-4">
          {loading && (
            <div className="card flex items-center justify-center py-20 text-gray-400 text-sm">
              <span className="animate-pulse">{t("inference.classifying")}</span>
            </div>
          )}

          {error && (
            <div className="card border-red-800 bg-red-950/30 text-red-400 text-sm">
              {error}
            </div>
          )}

          {!loading && !result && !error && (
            <div className="card flex items-center justify-center py-20 text-gray-600 text-sm border-dashed">
              {t("inference.empty_result")}
            </div>
          )}

          {result && !loading && (
            <div className="card space-y-6">
              {/* Top-1 강조 */}
              {top1 && (
                <div className="flex items-center gap-3 p-4 bg-brand-600/10 rounded-lg border border-brand-600/30">
                  <span className="text-3xl font-bold text-brand-400">
                    {(top1.confidence * 100).toFixed(1)}%
                  </span>
                  <div>
                    <p className="font-semibold text-lg text-white">
                      {top1.display_name ?? top1.character.replaceAll("_", " ")}
                    </p>
                    <p className="text-xs text-gray-500">{t("inference.best_match")}</p>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg bg-gray-900 border border-gray-800 p-3">
                  <p className="text-[10px] uppercase tracking-wider text-gray-500">
                    {t("inference.confidence_label")}
                  </p>
                  <p className="mt-1 text-lg font-semibold text-white tabular-nums">
                    {top1 ? top1.confidence.toFixed(4) : "-"}
                  </p>
                </div>
                <div className="col-span-2 rounded-lg bg-gray-900 border border-gray-800 p-3">
                  <p className="text-[10px] uppercase tracking-wider text-gray-500">
                    {t("inference.confidence_hint_title")}
                  </p>
                  <p className="mt-1 text-xs leading-5 text-gray-400">
                    {t("inference.confidence_hint")}
                  </p>
                </div>
              </div>

              {/* Top-5 바 */}
              <div className="space-y-3">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">{t("inference.top_k")}</p>
                {result.top_k.map((p) => (
                  <div key={p.rank} className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-400">
                      <span className={p.rank === 1 ? "text-gray-200 font-medium" : ""}>
                        {p.rank}. {p.display_name ?? p.character.replaceAll("_", " ")}
                      </span>
                      <span>{(p.confidence * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          p.rank === 1 ? "bg-brand-500" : "bg-gray-600"
                        }`}
                        style={{ width: `${p.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-gray-200">{t("inference.api_example_title")}</p>
            <p className="text-xs text-gray-500 mt-0.5">{t("inference.api_example_desc")}</p>
          </div>
          <code className="rounded-md bg-gray-950 border border-gray-800 px-2 py-1 text-[11px] text-brand-300">
            POST /api/inference/predict
          </code>
        </div>
        <pre className="max-h-80 overflow-auto rounded-lg bg-gray-950 border border-gray-800 p-4 text-xs leading-5 text-gray-300">
          <code>{apiExampleJson}</code>
        </pre>
      </div>
    </div>
  );
}
