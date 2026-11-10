"""Create an isolated Python 3.10 environment; leave Colab's kernel untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(argv, **kwargs):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=Path, default=Path("/content/sprig-py310"))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from check_colab_gpu import check_gpu
    check_gpu(require_colab=False)
    lock = json.loads((root / "colab/runtime-lock.json").read_text())
    for dep in lock["code"]:
        target = root / dep["path"]
        if not target.exists():
            target.mkdir(parents=True)
            run(["git", "init", str(target)])
            run(["git", "-C", target, "remote", "add", "origin", dep["url"]])
            run(["git", "-C", target, "fetch", "--depth", "1", "origin", dep["commit"]])
            run(["git", "-C", target, "checkout", "--detach", "FETCH_HEAD"])
        current = subprocess.check_output(["git", "-C", target, "rev-parse", "HEAD"], text=True).strip()
        if current != dep["commit"]:
            raise RuntimeError(f"Unexpected dependency revision at {target}; use a fresh runtime.")
    # Both original encoders use cwd-relative third_partys paths.
    link = root / "skinning/third_partys/Michelangelo"
    target = root / "skeleton/third_partys/Michelangelo"
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target.resolve():
        raise RuntimeError("Unexpected Michelangelo path; use a fresh runtime.")
    run([sys.executable, "-m", "pip", "install", "uv==0.9.5"])
    py = a.env / "bin/python"
    if not py.exists():
        run([sys.executable, "-m", "uv", "venv", "--python", lock["python"], "--seed", a.env])
    run([py, "-m", "pip", "install", "setuptools==69.5.1", "wheel==0.44.0", "cython==0.29.36", "ninja==1.11.1.1", "packaging==24.2"])
    run([py, "-m", "pip", "install", "torch==2.1.1", "torchvision==0.16.1", "torchaudio==2.1.1", "--index-url", "https://download.pytorch.org/whl/cu118"])
    run([py, "-m", "pip", "install", "-r", root / "colab/requirements-colab.txt"])
    # Use the official wheel, avoiding compilation against Colab's changing nvcc.
    run([py, "-m", "pip", "install", "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"])
    run([py, "-m", "pip", "install", "torch-scatter==2.1.2", "--no-index", "--find-links", "https://data.pyg.org/whl/torch-2.1.1+cu118.html"])
    run([py, "-m", "pip", "install", "pytorch3d", "--no-index", "--no-cache-dir", "--find-links", "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html"])
    run([py, "-m", "pip", "check"])
    run([py, root / "colab/check_runtime.py"])
    frozen = subprocess.check_output([str(py), "-m", "pip", "freeze"], text=True)
    (a.env / "installed-packages.txt").write_text(frozen)
    print("Runtime imports and a small FlashAttention CUDA operation passed. Full SPRig inference is a separate check.")


if __name__ == "__main__":
    main()
