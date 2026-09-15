"""GPU capability probes — which vendor path is available on this machine.

Vendor decides the engine: WhisperX accelerates on **NVIDIA CUDA only**
(CTranslate2/faster-whisper has no ROCm backend — an AMD GPU gets CPU speeds),
while **whisper.cpp reaches AMD and Intel through Vulkan**, which is
vendor-neutral. Probing lives here so the backend picker, the CLI and the
tests can share it without importing each other.

Leaf module: stdlib only.
"""

import os
import shutil
import subprocess

VULKAN_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")


def nvidia_gpu() -> str | None:
    """Name of the first NVIDIA GPU, or None (no nvidia-smi / no device)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0] if names else None


def vulkan_icds() -> list[str]:
    """Installed Vulkan driver manifests (ICDs) — one per GPU driver found.

    VK_ICD_FILENAMES / VK_DRIVER_FILES override the search path, matching the
    loader's own precedence, so a user with a custom driver layout still counts.
    """
    env = os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES")
    if env:
        return [p for p in env.split(os.pathsep) if p]
    found = []
    for directory in VULKAN_ICD_DIRS:
        try:
            found.extend(
                os.path.join(directory, name)
                for name in sorted(os.listdir(directory))
                if name.endswith(".json")
            )
        except OSError:
            continue
    return found


def vulkan_gpu_driver() -> str | None:
    """Vendor hint from the ICD manifest filenames (radeon/intel/nvidia/lvp)."""
    names = " ".join(os.path.basename(p).lower() for p in vulkan_icds())
    for vendor, marker in (
        ("AMD", "radeon"),
        ("AMD", "amdvlk"),
        ("Intel", "intel"),
        ("NVIDIA", "nvidia"),
        ("software", "lvp"),
        ("software", "lavapipe"),
    ):
        if marker in names:
            return vendor
    return None
