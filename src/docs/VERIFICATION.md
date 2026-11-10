# Verification performed during preparation

- 59 Python source files passed syntax parsing.
- The six untouched local files match their uploaded bytes. The seventh has only the documented cache portability patch.
- The local canonical-anchor helper passed synthetic natural-sort, max-area, and neighbor-order checks.
- Checkpoint export passed synthetic bare/model/state_dict round trips using fp32, bfloat16, and integer tensors, with all non-model fields excluded.
- Split validation rejected sequence duplicates, sequence overlap, and parent-asset overlap.
- Evaluation support imports and synthetic rigid-inverse/RMSE checks passed.
- Source packaging excludes unlisted dataset files and rejects changed hashes and unsafe paths.

These helper tests used CPU PyTorch 2.5.1+cpu, NumPy 1.26.4, and trimesh 4.2.3.
NumPy 2.x did not support the inherited trimesh 4.2.3 helper; the documented 1.26.4 pin was applied before the passing run.
They do not validate the complete historical CUDA model environment.
The original helper tests used no dataset or real checkpoint. Subsequent HF checks below use the actual uploaded checkpoints; no backbone forward, training or paper metric rerun was performed.

## Colab release additions

- Notebook validates against the nbformat schema; all code cells parse and all saved outputs are empty.
- The notebook fetches GitHub main, checks a fixed manifest hash and every reviewed source file, and records the checkout commit before using `src/`.
- Synthetic GPU detection cases admit full A100/H100/H200/GH200 devices and reject CPU, T4, L4, V100, A30, consumer GPUs, insufficient memory and MIG partitions. Blackwell gets a specific software-compatibility error.
- Strict loading accepts raw/model/state_dict tensor packages and rejects missing keys, extra keys and wrong shapes (CPU test tensors only).
- Generated-box mesh checks and synthetic skeleton/skin checks reject invalid trees, nonfinite values, wrong dimensions and unnormalized rows.
- The source ZIP has exactly the manifest's reviewed files plus the manifest, with matching hashes and no data/model binaries.

The actual files at wrp/SPRig commit e568ae8039dbd461cf7b0cdcc3220268bb0ec4d4 were downloaded completely. Both sizes and SHA-256 values match the HF LFS metadata. CPU PyTorch 2.5.1 weights_only=True successfully deserializes both: skeleton is a raw tensor state dictionary (701 entries); skinning contains a model tensor dictionary (596 entries), epoch, global_step, val_l1 and args. The loader selects the model tensors and ignores the auxiliary fields. This is serialization verification, not strict loading into an instantiated SPRig model.

Colab was not signed in and no local CUDA GPU was available. Conda reconstruction and GPU validation are paused at the author's request. The isolated CUDA environment installer and full two-stage inference remain unvalidated; passing hardware detection or CPU deserialization does not certify them.

## GitHub and default-example update

The notebook no longer requests source/OBJ uploads or a Drive mount. The default input is
Puppeteer's deer OBJ, downloaded from a pinned upstream revision with size/SHA-256 verification.
Bootstrap and mesh-preparation checks are CPU/network checks, not GPU inference validation.

The updated source has 60 Python files with valid syntax. The notebook passes nbformat
validation with empty outputs and contains no upload or Drive-mount calls. The exact upstream
deer OBJ was downloaded and verified (`0e7d8b75751aeead9ad217072d24323c127350567f9aa6bc86a5f5234feeec9c`),
then passed the existing mesh preparation code: 3,669 vertices and 4,131 triangles in the
canonical output. All 45 source-provenance records still match the selected code bytes.
These checks did not instantiate either neural network or run GPU inference.
