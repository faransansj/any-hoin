"""Intel Arc XPU / IPEX 호환성 유틸리티.

모든 XPU·IPEX 관련 코드를 한 곳에서 관리합니다.
이 파일을 torch 보다 먼저 import 하면 triton 마스킹이 자동 적용됩니다.
각 함수는 예외를 내부에서 처리하며 절대 크래시하지 않습니다.

사용법:
    import xpu_compat          # train.py / model_loader.py 등 최상단에서
    import torch               # 그 다음에 torch import
"""

from contextlib import contextmanager
import sys
import warnings

# triton-xpu 패키지는 triton.language 등 표준 API를 구현하지 않아
# torch._dynamo / IPEX import 시 AttributeError 를 일으킨다.
# None 으로 선점해 triton-free 경로를 강제한다.
# sys.modules 에 이미 있으면 setdefault 는 아무 일도 하지 않는다.
sys.modules.setdefault("triton", None)  # type: ignore[arg-type]

# ──────────────────────────────────────────────────────────────────
# IPEX lazy-load (캐시)
# ──────────────────────────────────────────────────────────────────

_ipex_module = None   # intel_extension_for_pytorch 또는 None
_ipex_loaded = False  # 로드 시도 완료 여부


class _NoopGradScaler:
    """torch GradScaler API와 맞춘 CPU/no-op 폴백."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict) -> None:
        return None


def _load_ipex() -> bool:
    """IPEX를 한 번만 로드하고 결과를 캐시. 실패 시 False 반환."""
    global _ipex_module, _ipex_loaded
    if _ipex_loaded:
        return _ipex_module is not None
    _ipex_loaded = True
    try:
        import torch
        if "+xpu" not in torch.__version__:
            return False

        from importlib.metadata import PackageNotFoundError, version as _pkg_ver
        try:
            ipex_ver = _pkg_ver("intel-extension-for-pytorch")
        except PackageNotFoundError:
            return False

        # major.minor 불일치 시 IPEX __init__.py 가 os._exit(127)를 호출하므로
        # import 전에 반드시 확인해야 한다.
        torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        ipex_mm  = ".".join(ipex_ver.split("+")[0].split(".")[:2])
        if torch_mm != ipex_mm:
            print(
                f"[XPU] IPEX {ipex_ver}는 torch {ipex_mm}.x용이지만 "
                f"현재 torch {torch.__version__}가 설치되어 있습니다. "
                "IPEX 최적화를 건너뜁니다."
            )
            return False

        import intel_extension_for_pytorch as _m
        _ipex_module = _m
        return True
    except Exception as exc:
        print(f"[XPU] IPEX 로드 실패: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────

def torch_version() -> str:
    """설치된 torch 버전. torch import 실패 시 빈 문자열."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return ""


def is_xpu_build() -> bool:
    """현재 PyTorch가 XPU 빌드인지 안전하게 확인."""
    return "+xpu" in torch_version()


def xpu_available() -> bool:
    """XPU 빌드이고 하드웨어가 실제로 존재하는지 안전하게 확인."""
    try:
        import torch
        if "+xpu" not in torch.__version__:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return bool(torch.xpu.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    """CUDA 사용 가능 여부."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    """Apple MPS 사용 가능 여부."""
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def xpu_unavailable_message(option: str) -> str:
    """CLI/UI에서 공통으로 쓰는 XPU 사용 불가 사유."""
    version = torch_version() or "unknown"
    if "+xpu" not in version:
        return (
            f"{option} 를 지정했지만 현재 PyTorch는 XPU 빌드가 아닙니다.\n"
            f"  - installed torch: {version}\n"
            "  - XPU 학습에는 '+xpu' PyTorch 빌드가 필요합니다.\n"
            "  - Intel Arc를 사용하려면: uv sync --extra arc\n"
            "  - NVIDIA/CUDA 빌드를 쓰는 환경에서는 device를 auto/cuda/cpu로 선택하세요."
        )
    return (
        f"{option} 를 지정했지만 torch.xpu.is_available() == False 입니다.\n"
        "  1) Intel GPU 드라이버/Level Zero 설치 확인\n"
        "  2) 현재 사용자가 render/video 그룹에 포함됐는지 확인\n"
        "  3) uv sync --extra arc 로 XPU 빌드를 설치했는지 확인\n"
        "  4) docs/intel_arc_setup.md 참조"
    )


def xpu_status() -> dict:
    """UI 표시용 XPU 상태."""
    version = torch_version() or "unknown"
    build = "+xpu" in version
    available = xpu_available()
    reason = None
    if not build:
        reason = (
            f"현재 torch {version}는 XPU 빌드가 아닙니다. "
            "Intel Arc 사용 시 uv sync --extra arc가 필요합니다."
        )
    elif not available:
        reason = "torch.xpu.is_available() == False 입니다. 드라이버/Level Zero/render 그룹을 확인하세요."
    return {"build": build, "available": available, "reason": reason}


def device_available(device_type: str) -> bool:
    """디바이스 타입별 사용 가능 여부."""
    if device_type == "xpu":
        return xpu_available()
    if device_type == "cuda":
        return cuda_available()
    if device_type == "mps":
        return mps_available()
    if device_type == "cpu":
        return True
    return False


def require_device_available(device, option: str) -> None:
    """명시 요청된 디바이스가 없으면 RuntimeError."""
    dev_type = getattr(device, "type", str(device).split(":", 1)[0])
    if dev_type == "xpu" and not xpu_available():
        raise RuntimeError(xpu_unavailable_message(option))
    if dev_type == "cuda" and not cuda_available():
        raise RuntimeError(
            f"{option} 를 지정했지만 CUDA를 사용할 수 없습니다.\n"
            f"  - installed torch: {torch_version() or 'unknown'}\n"
            "  - NVIDIA GPU/드라이버/CUDA PyTorch 빌드를 확인하거나 auto/cpu를 선택하세요."
        )
    if dev_type == "mps" and not mps_available():
        raise RuntimeError(
            f"{option} 를 지정했지만 Apple MPS를 사용할 수 없습니다. auto/cpu를 선택하세요."
        )


def ipex_available() -> bool:
    """IPEX 사용 가능 여부 (lazy, 캐시)."""
    return _load_ipex()


def ipex_version() -> str | None:
    """설치된 IPEX 패키지 버전."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("intel-extension-for-pytorch")
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def try_ipex_optimize(model, optimizer, *, use_amp: bool):
    """ipex.optimize() 시도.

    성공하면 최적화된 (model, optimizer) 반환.
    실패 시 경고만 출력하고 원본을 그대로 반환 — 크래시 없음.
    백본이 ConvNeXt / EfficientNet 등 IPEX 미지원 구조여도 안전.
    """
    if not _load_ipex():
        return model, optimizer
    try:
        import torch
        dtype = torch.bfloat16 if use_amp else torch.float32
        return _ipex_module.optimize(model, optimizer=optimizer, dtype=dtype)
    except Exception as exc:
        print(f"[XPU] ipex.optimize() 실패 — 최적화 없이 계속합니다: {exc}")
        return model, optimizer


def make_grad_scaler(device_type: str, enabled: bool):
    """device-aware GradScaler.

    XPU / CUDA / 기타 순으로 시도하고, 지원하지 않으면 CPU no-op으로 폴백.
    """
    try:
        import torch
    except Exception:
        return _NoopGradScaler()

    def _disabled():
        try:
            return torch.amp.GradScaler("cpu", enabled=False)
        except Exception:
            return _NoopGradScaler()

    if device_type == "xpu":
        try:
            return torch.amp.GradScaler("xpu", enabled=enabled)
        except Exception as exc:
            if enabled:
                print(f"[XPU] GradScaler 사용 불가 — AMP scaler를 비활성화합니다: {exc}")
            return _disabled()
    if device_type == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except Exception as exc:
            if enabled:
                print(f"[CUDA] GradScaler 사용 불가 — AMP scaler를 비활성화합니다: {exc}")
            return _disabled()
    return _disabled()


@contextmanager
def autocast(device_type: str, enabled: bool):
    """AMP autocast context. 생성/진입 실패 시 FP32로 계속 진행."""
    if not enabled:
        yield
        return

    try:
        import torch
        ctx = torch.amp.autocast(device_type=device_type, enabled=True)
        ctx.__enter__()
    except Exception as exc:
        print(f"[{device_type.upper()}] autocast 사용 불가 — FP32로 계속합니다: {exc}")
        yield
        return

    try:
        yield
    except BaseException as exc:
        suppress = ctx.__exit__(type(exc), exc, exc.__traceback__)
        if not suppress:
            raise
    else:
        ctx.__exit__(None, None, None)


def best_device() -> "torch.device":
    """XPU → CUDA → MPS → CPU 순으로 사용 가능한 디바이스 반환."""
    import torch
    if xpu_available():
        return torch.device("xpu")
    if cuda_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device) -> str:
    """사람이 읽기 좋은 디바이스 설명."""
    try:
        import torch
        dev_type = getattr(device, "type", str(device).split(":", 1)[0])
        if dev_type == "xpu":
            try:
                name = torch.xpu.get_device_name(0)
            except Exception:
                name = "Intel XPU"
            return f"Intel Arc GPU (XPU) — {name}"
        if dev_type == "cuda":
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA GPU"
            hip = getattr(torch.version, "hip", None)
            if hip is not None:
                return f"AMD ROCm GPU — {name} (HIP {hip})"
            return f"NVIDIA CUDA GPU — {name} (CUDA {torch.version.cuda})"
        if dev_type == "mps":
            return "Apple Silicon GPU (MPS)"
    except Exception:
        pass
    return "CPU"
