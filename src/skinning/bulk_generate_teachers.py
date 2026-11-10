#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supplementary skinning-teacher generation script for SPRig.

This script prepares canonical-anchor teacher skinning data for each
DeformingThings4D sequence. For every sequence, it selects a canonical
anchor frame, copies the corresponding mesh and skeleton into a temporary
generation folder, invokes the original Puppeteer skinning pipeline, and
stores the resulting canonical skinning files under a sequence-specific
directory.

Expected output layout:
    <teacher_root>/<sequence_name>/
        <sequence_name>_anchor_XXXX_skin.txt
        <sequence_name>_anchor_XXXX_skin.npy
        anchor_meta.json
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from anchor_discovery import find_canonical_anchor, list_mesh_files


def find_anchor_frame(seq_dir: Path, criterion: str = "area"):
    """
    Return the canonical anchor frame for a sequence.

    Returns:
        anchor_path: Path to the selected anchor mesh.
        anchor_idx:  Index of the selected frame in the naturally sorted list.
        anchor_score: Heuristic score used for anchor selection.
    """
    files = list_mesh_files(seq_dir)
    idx, score, _ = find_canonical_anchor(seq_dir, criterion=criterion, files=files)
    return files[idx], int(idx), float(score)


def count_existing_pairs(teacher_root: Path):
    """
    Count sequences that already contain a generated skinning pair.

    A valid pair is defined as the co-existence of:
        *_skin.txt and *_skin.npy
    under a sequence subdirectory.
    """
    ok = 0
    seqs_ok = []
    for seq_dir in sorted([d for d in teacher_root.iterdir() if d.is_dir()]):
        seq = seq_dir.name
        txts = list(seq_dir.glob(f"{seq}_*_skin.txt"))
        npys = list(seq_dir.glob(f"{seq}_*_skin.npy"))
        if txts and npys:
            ok += 1
            seqs_ok.append(seq)
    return ok, seqs_ok


def main():
    ap = argparse.ArgumentParser(
        description="Generate canonical-anchor teacher skinning files for SPRig post-training."
    )
    ap.add_argument("--dt4d_root", type=str, default="../DeformingThings4D_objs")
    ap.add_argument("--skel_folder", type=str, default="../skeleton/DeformingThings4D_skeleton_txt")
    ap.add_argument("--pretrained_weights", type=str, required=True)
    ap.add_argument("--python_bin", type=str, default="python")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--post_filter", action="store_true", default=True)
    ap.add_argument(
        "--anchor_criterion",
        type=str,
        default="area",
        choices=["area", "bbox_volume"],
        help="Heuristic used to select the canonical anchor frame.",
    )
    ap.add_argument(
        "--teacher_root",
        type=str,
        default="./outputs_txt",
        help="Root directory used to store canonical teacher skinning outputs.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dt4d_root = Path(args.dt4d_root)
    skel_folder = Path(args.skel_folder)
    teacher_root = Path(args.teacher_root)
    teacher_root.mkdir(parents=True, exist_ok=True)

    count_existing_pairs(teacher_root)

    all_seqs = [d.name for d in sorted(dt4d_root.iterdir()) if d.is_dir()]

    tmp_mesh = Path("./temp_meshes_for_gen")
    tmp_skel = Path("./temp_skels_for_gen")
    tmp_mesh.mkdir(exist_ok=True)
    tmp_skel.mkdir(exist_ok=True)

    todo = list(all_seqs)

    pbar = tqdm(todo, desc="Generating canonical teacher skinning", unit="seq")
    for seq in pbar:
        seq_dir = dt4d_root / seq
        rig_txt = skel_folder / f"{seq}.txt"
        if not rig_txt.exists():
            continue

        try:
            anchor_obj, anchor_idx, anchor_score = find_anchor_frame(
                seq_dir,
                criterion=args.anchor_criterion,
            )
        except Exception:
            continue

        anchor_tag = f"anchor_{anchor_idx:04d}"

        seq_out_dir = teacher_root / seq
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        want_txt = seq_out_dir / f"{seq}_{anchor_tag}_skin.txt"
        want_npy = seq_out_dir / f"{seq}_{anchor_tag}_skin.npy"

        if want_txt.exists() and want_npy.exists() and not args.overwrite:
            continue

        for p in tmp_mesh.glob("*"):
            p.unlink()
        for p in tmp_skel.glob("*"):
            p.unlink()

        shutil.copy(str(anchor_obj), tmp_mesh / f"{seq}_{anchor_tag}.obj")
        shutil.copy(str(rig_txt), tmp_skel / f"{seq}_{anchor_tag}.txt")

        cmd = [
            args.python_bin,
            "main.py",
            "--num_workers",
            "1",
            "--batch_size",
            "1",
            "--generate",
            "--save_skin_npy",
            "--pretrained_weights",
            args.pretrained_weights,
            "--input_skel_folder",
            str(tmp_skel),
            "--mesh_folder",
            str(tmp_mesh),
            "--depth",
            str(args.depth),
            "--save_folder",
            str(seq_out_dir),
        ]
        if args.post_filter:
            cmd.append("--post_filter")

        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        gen_dir = seq_out_dir / "generate"
        gen_txt = gen_dir / f"{seq}_{anchor_tag}_skin.txt"
        gen_npy = gen_dir / f"{seq}_{anchor_tag}_skin.npy"

        if not gen_txt.exists() or not gen_npy.exists():
            continue

        shutil.move(str(gen_txt), str(want_txt))
        shutil.move(str(gen_npy), str(want_npy))

        try:
            if gen_dir.exists() and not any(gen_dir.iterdir()):
                gen_dir.rmdir()
        except Exception:
            pass

        try:
            (seq_out_dir / "anchor_meta.json").write_text(
                json.dumps(
                    {
                        "sequence_id": seq,
                        "anchor_idx": anchor_idx,
                        "anchor_score": anchor_score,
                        "anchor_mesh_name": anchor_obj.name,
                        "anchor_criterion": args.anchor_criterion,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
