# Reproducibility status

The Colab inference entry point uses GitHub source under `src/`, the author's confirmed
skeleton `DeformingThings4D_finetune_v2` and skinning `finetuned_model_v3`, and Puppeteer's
public deer example. Source content is pinned by manifest SHA-256; weights, encoder prerequisites and example revisions are pinned.
The example and the two SPRig checkpoints have size and SHA-256 guards.

The revised uploaded local source takes precedence over the older Drive copies.
Missing support modules come from the working Drive project; `SOURCE_SELECTION.json`
records the source hashes. The Stage-1 trainer's global Colab cache deletion was removed
for portability. The training objectives were not changed during release preparation.

Both actual HF checkpoint files have passed full-file integrity and CPU deserialization
checks. This is not strict loading into instantiated models and is not a forward-pass test.
The fresh Colab A100 environment and the complete skeleton-to-skinning run remain unvalidated.
The isolated Python/CUDA setup is a candidate, not a reconstruction of the historical Conda env.

This demo performs inference only. It does not rerun training or paper metrics, generate
animation, or redistribute the paper datasets. See `VERIFICATION.md` for the scope of checks.
