"""Run skeleton and skinning in separate processes to release GPU memory between stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from check_colab_gpu import check_gpu
from demo_io import prepare_mesh, validate_rig, validate_skin


def run_logged(command, cwd, env, log):
    with log.open("w") as f:
        proc = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True); f.write(line); f.flush()
        if proc.wait() != 0:
            raise RuntimeError(f"Inference stage failed; see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="selected_weights.json")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    gpu = check_gpu(require_colab=False)
    weights = json.loads(a.weights.read_text())
    for key in ("skeleton", "skinning"):
        if not Path(weights[key]).is_file():
            raise FileNotFoundError(weights[key])
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory to prevent stale results: {out}")
    out.mkdir(parents=True)
    for folder in ("meshes", "skeleton", "skinning", "logs"):
        (out / folder).mkdir()
    mesh = out / "meshes/mesh.obj"
    report = {"status": "running", "gpu": gpu, "source": weights.get("source"), "seed": 0,
              "paper_metrics_reproduced": False, "input_sha256": hashlib.sha256(a.mesh.read_bytes()).hexdigest()}
    start = time.time()
    try:
        nv, nf = prepare_mesh(a.mesh, mesh)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(x) for x in [root / "skeleton/third_partys/Michelangelo", root / "skinning/third_partys/PartField", root / "skeleton", root / "skinning"])
        env["PYTHONUNBUFFERED"] = "1"
        run_logged([sys.executable, root / "skeleton/infer_release.py", "--mesh", mesh,
                    "--checkpoint", weights["skeleton"], "--llm", weights["llm_config"],
                    "--output", out / "skeleton"], root / "skeleton", env, out / "logs/skeleton.log")
        nj = validate_rig(out / "skeleton/mesh.txt")
        run_logged([sys.executable, root / "skinning/infer_release.py", "--generate",
                    "--pretrained_weights", weights["skinning"], "--mesh_folder", out / "meshes",
                    "--input_skel_folder", out / "skeleton", "--save_folder", out / "skinning",
                    "--batch_size", "1", "--num_workers", "0", "--depth", "1", "--max_joints", "70"],
                   root / "skinning", env, out / "logs/skinning.log")
        shape = validate_skin(out / "skinning/generate/mesh_skin.npy", nv, nj)
        validate_rig(out / "skinning/generate/mesh_skin.txt")
        report.update(status="passed", vertices=nv, faces=nf, joints=nj, skin_shape=shape,
                      elapsed_seconds=round(time.time() - start, 3))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (out / "run_report.json").write_text(json.dumps(report, indent=2))
    shutil.make_archive(str(out), "zip", out)
    print(f"Full inference completed. Result archive: {out}.zip")


if __name__ == "__main__":
    main()
