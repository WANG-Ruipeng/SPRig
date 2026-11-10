"""Release hardware policy, checked before installation or model downloads.

This is a distribution requirement, not a measured minimum for inference.
No torch import: this also runs in Colab's original Python environment.
"""
import csv
import importlib.util
import io
import json
import re
import subprocess

SUPPORTED = ("A100", "H100", "H200", "GH200")
NEWER_PENDING = ("B100", "B200", "B300", "GB200", "GB300")
MIN_MEMORY_MIB = 38 * 1024  # Allows the reporting difference for a nominal 40 GB A100.
HELP = "Colab: Runtime > Change runtime type > GPU > A100. CPU, T4, L4 and V100 are not supported by this release."


def evaluate_gpu(name, memory_mib, compute_capability, mig_listing=""):
    if "MIG" in name.upper() or re.search(r"\bMIG\s+\d+g", mig_listing):
        raise RuntimeError("A full GPU is required; MIG partitions are not supported. " + HELP)
    family = next((x for x in SUPPORTED + NEWER_PENDING
                   if re.search(r"(?<![A-Z0-9])" + x + r"(?![A-Z0-9])", name.upper())), None)
    if family is None:
        raise RuntimeError(f"Detected {name}. Requires NVIDIA A100 or a supported higher-tier GPU. " + HELP)
    if float(memory_mib) < MIN_MEMORY_MIB:
        raise RuntimeError(f"Detected only {float(memory_mib)/1024:.1f} GiB. A full A100 40/80 GB or supported higher-tier GPU is required.")
    cc = tuple(map(int, str(compute_capability).strip().split(".")))
    if cc < (8, 0):
        raise RuntimeError(f"Unsupported CUDA compute capability {compute_capability}. " + HELP)
    if family in NEWER_PENDING or cc >= (10, 0):
        raise RuntimeError(f"{name} meets the hardware tier, but this legacy CUDA 11.8 environment has no validated Blackwell profile. Select A100/H100/H200 for this notebook; do not bypass the check.")
    return {"name": name, "family": family, "memory_mib": float(memory_mib),
            "compute_capability": str(compute_capability), "hardware_policy": "passed",
            "end_to_end_validation": "not_implied"}


def check_gpu(require_colab=True):
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        in_colab = False
    if require_colab and not in_colab:
        raise RuntimeError("Open SPRig_Colab_Demo.ipynb in Google Colab. " + HELP)
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], text=True)
        listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("No usable NVIDIA GPU was detected. " + HELP) from exc
    rows = list(csv.reader(io.StringIO(raw.strip())))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError("This Colab demo expects exactly one full NVIDIA GPU.")
    result = evaluate_gpu(*(x.strip() for x in rows[0]), mig_listing=listing)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    check_gpu(require_colab=False)
