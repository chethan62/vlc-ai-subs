"""
Core device module — CUDA runtime preload + Whisper device/compute-type resolution.

The preload step makes faster-whisper (ctranslate2) find CUDA libraries even
when VLC launches the backend without LD_LIBRARY_PATH set. It's a no-op on
CPU-only systems.
"""

import ctypes
import os
from ctypes.util import find_library as _ctypes_find_library


def _cuda_lib_search_dirs() -> list[str]:
    """Return directories likely to contain CUDA runtime shared libraries."""
    dirs: list[str] = []
    # CUDA_HOME / CUDA_PATH
    for var in ("CUDA_HOME", "CUDA_PATH"):
        base = os.environ.get(var)
        if base:
            dirs += [os.path.join(base, "lib"), os.path.join(base, "lib64")]
    # Well-known system prefixes
    for prefix in ("/usr/local/cuda", os.path.expanduser("~/.local/cuda12")):
        dirs += [os.path.join(prefix, "lib"), os.path.join(prefix, "lib64")]
    # Distribution-specific
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/wsl/lib", "/opt/cuda/lib64"):
        if os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "libcublas.so.12"))
            or os.path.exists(os.path.join(d, "libcublas.so.13"))
        ):
            dirs.append(d)
    return dirs


def _preload_cuda_libs() -> bool:
    """Load CUDA runtime shared libs (cublas, cudart, etc.) via ctypes.

    Returns True if at least one lib was successfully loaded.
    """
    if os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower() == "cpu":
        return False

    wanted = ["libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12", "libnvblas.so.12"]

    def candidate_paths(name: str):
        for libdir in _cuda_lib_search_dirs():
            yield os.path.join(libdir, name)
        p = _ctypes_find_library(name.partition(".")[0])
        if p:
            yield p

    loaded_any = False
    for name in wanted:
        for path in candidate_paths(name):
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path)
                    loaded_any = True
                    break
                except OSError:
                    continue
    return loaded_any


def detect_device() -> tuple[str, str]:
    """Resolve (device, compute_type) for faster-whisper.

    Priority: VSCL_AISUBS_DEVICE env → auto-detect CUDA → CPU.
    On GPU the best compute type is int8_float16 (small VRAM footprint).
    """
    _preload_cuda_libs()

    env_device = os.environ.get("VSCL_AISUBS_DEVICE", "").strip().lower()
    if env_device:
        device = env_device
    else:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    env_ct = os.environ.get("VSCL_AISUBS_COMPUTE", "").strip().lower()
    if env_ct:
        compute = env_ct if env_ct != "default" else "default"
    else:
        compute = "int8_float16" if device == "cuda" else "float32"

    return device, compute
