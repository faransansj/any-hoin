# Agent Guidelines

## Model Export Packages

When adding or changing model export behavior, preserve the deployment package contract:

- Every deployable model folder or ZIP must include `class_map.json`.
- Every deployable PyTorch model must include `config.json` so the inference loader can rebuild the correct backbone and class count.
- Every deployable ONNX model that references external data must include the referenced `best_model.onnx.data` file next to `best_model.onnx`.
- `class_map.json` is model-specific. Do not reuse a class map across models unless the output class count and class order are known to match.
- The expected deployment folder shape is:

```text
models/<model-name>/
  best_model.pth
  best_model_fp16.pth
  best_model.onnx
  best_model.onnx.data
  class_map.json
  config.json
```

Optional quantized files such as `best_model_int8.pth`, `best_model_int4.pth`, and `best_model_int2.pth` may be included when present.

The Studio Export page should expose a single deployment ZIP download that packages the available model files together with required metadata. Do not add model-only downloads as the primary deployment path, because model weights without `class_map.json` are not usable for inference.
