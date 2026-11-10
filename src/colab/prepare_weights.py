"""Download explicit checkpoints/configs only; never download datasets."""
import argparse
import json
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from check_file_integrity import verify_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="")
    p.add_argument("--revision", default="")
    p.add_argument("--local-skeleton", type=Path)
    p.add_argument("--local-skinning", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "colab/runtime-lock.json").read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    if a.local_skeleton or a.local_skinning:
        if not (a.local_skeleton and a.local_skinning and a.local_skeleton.is_file() and a.local_skinning.is_file()):
            raise ValueError("Provide both existing local model checkpoint files.")
        selected = {"skeleton": str(a.local_skeleton.resolve()), "skinning": str(a.local_skinning.resolve()), "source": "local_author_validation"}
    else:
        release = lock['paper_weights']
        repo = a.repo or release['hf_repo_id']
        revision = a.revision or release['hf_revision']
        if (repo, revision) != (release['hf_repo_id'], release['hf_revision']):
            raise ValueError("This source release is pinned to one reviewed model pair. Use its matching repository and commit.")
        selected = {}
        integrity = {}
        for stage in ('skeleton', 'skinning'):
            item = release[stage]
            path = hf_hub_download(repo, item['filename'], revision=revision,
                                   local_dir=a.output, token=False)
            integrity[stage] = verify_file(path, expected_size=item['size_bytes'], expected_sha256=item['sha256'])
            selected[stage] = path
            print(f"Verified {stage} checkpoint size and SHA-256.", flush=True)
        selected.update(source='huggingface', repo=repo, revision=revision, integrity=integrity)
    for item in lock["encoder_weights"] + [lock["llm_config"]]:
        hf_hub_download(item["repo"], item["file"], revision=item["revision"], local_dir=root / item["local_dir"])
    # The pinned PartField config embeds a repository-root-relative ckpt path.
    # Resolve it explicitly because skinning is launched from its own folder.
    from omegaconf import OmegaConf
    config = root / "skinning/third_partys/PartField/configs/final/skinning.yaml"
    cfg = OmegaConf.load(config)
    cfg.continue_ckpt = str(root / "skinning/third_partys/PartField/ckpt/model_objaverse.ckpt")
    OmegaConf.save(cfg, config)
    selected["llm_config"] = str(root / lock["llm_config"]["local_dir"])
    (a.output / "selected_weights.json").write_text(json.dumps(selected, indent=2))
    print("Selected the explicit skeleton/skinning pair and downloaded encoder prerequisites. No datasets downloaded.")


if __name__ == "__main__":
    main()
