"""Strict checkpoint loading for the existing single-GPU skinning generator."""
import sys
from pathlib import Path
import numpy as np
import torch
from main import get_args_parser, generate
from utils.skin_data import SkinData
from skinning_models.models import SkinningNetStacked

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from checkpoint_io import load_strict


def main():
    args = get_args_parser().parse_args()
    if not args.generate or not args.pretrained_weights:
        raise ValueError("Use --generate and an explicit --pretrained_weights checkpoint.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = SkinData(args, mode="generate", query_num=8192)
    if len(dataset) != 1:
        raise ValueError("Demo expects exactly one matching OBJ / skeleton TXT pair.")
    model = SkinningNetStacked(args)
    load_strict(model, args.pretrained_weights)
    model.cuda().eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    with torch.inference_mode():
        generate(loader, model, torch.device("cuda"), args)
    results = list((Path(args.save_folder) / "generate").glob("*_skin.npy"))
    if len(results) != 1:
        raise RuntimeError("Skinning did not produce exactly one output.")
    w = np.load(results[0], allow_pickle=False)
    if w.ndim != 2 or not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(1), 1, atol=1e-4):
        raise RuntimeError("Invalid skinning weights: expected finite, nonnegative, normalized rows.")
    print(f"Validated skinning output: {w.shape[0]} vertices × {w.shape[1]} joints")


if __name__ == "__main__":
    main()
