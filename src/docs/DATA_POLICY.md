# Data boundary for this release

The project owner requested a code-and-model-parameters release without redistribution
of any paper dataset. This document records that release choice; it does not claim that
all listed datasets have identical licenses.

## Include

Reviewed source code, documentation, configuration descriptions, source checksums, and
explicitly selected trained neural-network checkpoints (hosted separately on Hugging Face). Split identifiers may be added after their provenance and separation are checked.

## Exclude

Raw meshes/animations/textures; OBJ/PLY/FBX/GLB/BLEND assets; point clouds and sampled
surface coordinates; HDF5 benchmark files; NPZ packs; cached encoder features;
per-object skeleton pseudo-labels; teacher outputs; per-mesh skinning arrays; visualization
HTML with embedded points; dataset renders/videos; original notebooks and outputs;
temporary files, logs, full git history, backup scripts, and original model archives.

In the inspected project this includes `DeformingThings4D_objs/`, `Dymesh_objs/`,
`DeformingThings4D_datasets*/`, `DeformingThings4D_skeleton_txt*/`, `teacher_root*/`,
`triplet_npz_single/`, `output_npz/`, `outputs_txt*/`, `testset/`, `dataset_parts/`,
`examples/`, `assets/`, `results/`, `Demo*/`, and the three skinning benchmark `.h5` files.
Folder names are not a complete classifier: the source builder uses an explicit file allowlist.

## Obtain data from the original providers

- DT4D: https://github.com/rabbityl/DeformingThings4D
- DyMesh: https://github.com/JarrentWu1031/AnimateAnyMesh
- D-FAUST: https://dfaust.is.tue.mpg.de/
- Articulation-XL 2.0 / Diverse-Pose: https://huggingface.co/datasets/Seed3D/Articulation-XL2.0
- ModelsResource benchmark preparation: https://github.com/zhan-xu/RigNet

Users obtain the applicable access permissions and data independently. No dataset mirrors,
download credentials, or original private Drive folder links belong in the public README.

## Automatic demo input

The Colab notebook downloads only Puppeteer's official `examples/deer.obj` from a pinned
upstream commit. Its bytes are verified before use. The OBJ remains outside this source
release; this does not enable downloading or mirroring any paper dataset.
