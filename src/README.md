# SPRig implementation

Open [`SPRig_Colab_Demo.ipynb`](../SPRig_Colab_Demo.ipynb) in
[Google Colab](https://colab.research.google.com/github/WANG-Ruipeng/SPRig/blob/main/SPRig_Colab_Demo.ipynb),
select **A100**, and choose **Run all**. The notebook fetches this repository from GitHub and verifies a fixed source manifest,
uses `src/` as the code root, installs an isolated Python 3.10 environment, obtains checkpoints
from Hugging Face, and automatically downloads Puppeteer's official `examples/deer.obj`.
It predicts a skeleton and skinning, previews the result, and downloads a ZIP after successful checks.
No ZIP/OBJ upload, Drive mount, or Hugging Face login is required.

**Status: the GitHub/example bootstrap is connected; complete GPU inference remains unvalidated.**
The historical Conda environment has not been reconstructed or certified compatible.
The isolated installer is a candidate and must pass its import/CUDA checks before inference.

## Model pair confirmed by the author

| Stage | Selected training run | HF release filename |
|---|---|---|
| Skeleton | `DeformingThings4D_finetune_v2/checkpoints/best_model.pth` | `sprig_dt4d_skeleton.pth` |
| Skinning | `finetuned_model_v3/ckpts/best.pth` | `sprig_dt4d_skinning.pth` |

The run selection is author-confirmed. Both uploaded original files have been fully downloaded, checked against their HF SHA-256 values and deserialized with CPU PyTorch 2.5.1 using `weights_only=True`. Skeleton is a raw tensor dictionary; skinning wraps its tensors under `model` alongside training metadata. The inference loader supports both formats. This does not verify architecture compatibility or GPU inference: strict parameter loading will still stop on any missing, extra or mismatched model parameter.

## Hardware policy

The first notebook cell checks the GPU before installing dependencies or downloading weights. The current runtime profile targets A100, H100, H200 and GH200 with at least 38 GiB of reported total memory (allowing the reporting difference for a nominal 40 GB device). It rejects CPU, T4, L4, V100, A30, consumer GPUs and MIG partitions. Newer Blackwell models receive an explicit compatibility message because this CUDA 11.8 profile has not been adapted to them.

This is the requested release hardware policy, **not a measured minimum VRAM claim or a guarantee that every admitted GPU has been tested**. A100 is the initial validation target. Colab must actually allocate the selected GPU; the notebook cannot guarantee availability.

## Usage

1. Open the notebook in Colab, select A100 and run all cells.
2. Step 3 fetches GitHub `main`, verifies the reviewed source hashes, records the Git commit and uses `src/`.
3. Step 4 installs dependencies and runs CUDA/import checks; failures stop the notebook.
4. Step 5 obtains the public `wrp/SPRig` checkpoints at the prefilled revision and verifies their hashes.
5. Step 6 obtains Puppeteer's pinned deer OBJ, verifies its hash and runs skeleton and skinning in separate processes.
6. Step 7 previews and downloads the result only when its status is `passed`.

The result contains `meshes/mesh.obj`, `skeleton/mesh.txt`, `skinning/generate/mesh_skin.txt`, `mesh_skin.npy`, a preview JSON, logs and `run_report.json`. The skin array indexes the **exported `meshes/mesh.obj`**, whose vertex order can differ from the input OBJ. Skeletons outside 2–70 joints, invalid trees, wrong output dimensions, nonfinite values and unnormalized skin weights stop the pipeline. This demo does not generate motion or reproduce paper metrics.

The new `skeleton/infer_release.py` uses the revised local SPRig model. It does not use the old upstream `skeleton/demo.py` parameter-loading fallback. The two stages run in separate processes so that skeleton GPU memory is released before skinning.

## Preparing Hugging Face weights

Run in a trusted environment with the selected Drive checkpoints accessible:

```bash
python tools/export_weights.py --skeleton /path/to/DeformingThings4D_finetune_v2/checkpoints/best_model.pth \
  --skinning /path/to/finetuned_model_v3/ckpts/best.pth --output /path/to/new_export_dir
```

The original checkpoint files are already public at `https://huggingface.co/wrp/SPRig`. To replace them with tensor-only exports later, use this exporter, then update the pinned commit and hashes as one release change. The exporter excludes optimizer state and training paths and verifies tensor equality. Per-mesh skinning arrays are dataset derivatives, not neural-network checkpoint weights, and must not be uploaded as model weights.

## Source provenance and data exclusions

The author's revised uploaded code takes precedence over Drive versions. Missing support modules come from the working Drive project; see `SOURCE_SELECTION.json`. The revised training/evaluation files remain available for research, but the notebook invokes inference only. The training algorithm is unchanged; the previous global cache deletion was removed for portability.

No DT4D, DyMesh, D-FAUST, Articulation-XL, ModelsResource, Diverse-Pose dataset, processed data pack, pseudo-label pack, or dataset-derived skin array is included. The default example is downloaded directly from Puppeteer's pinned upstream revision; it is not bundled here. Upstream encoder code and pretrained weights are obtained from their original providers at runtime; their licenses remain applicable.

`colab/runtime-lock.json` pins upstream dependencies selected during preparation, not the unknown historical training checkout. Python/Torch/CUDA and key packages follow Puppeteer's documented environment; transitive packages are recorded by `pip freeze` after installation. The environment is a candidate until the actual GPU run succeeds.

`LICENSE_PUPPETEER.txt` preserves upstream attribution. It is not a newly chosen blanket license for SPRig contributions or weights. No additional license is inferred for SPRig contributions or model weights.

## Source packaging

```bash
python tools/build_source_release.py --output /tmp/SPRig_source.zip
```

Only reviewed source files in `RELEASE_MANIFEST.json` are included. Unknown datasets, weight files, run outputs and installed dependencies are excluded even if present locally. The maintenance bundle includes private audit records; **publish the source ZIP and notebook, not the maintenance bundle**.
