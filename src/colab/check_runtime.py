"""Fail before large checkpoint downloads if the CUDA extension stack is unusable."""
import importlib
import json
import sys
import torch
from flash_attn import flash_attn_func

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == "2.1.1+cu118", torch.__version__
assert torch.cuda.is_available(), "No CUDA device visible to the isolated Python environment"
for name in ("xformers", "torch_scatter", "pytorch3d", "mesh2sdf", "transformers", "lightning", "trimesh"):
    importlib.import_module(name)
q = torch.randn((1, 128, 8, 64), device="cuda", dtype=torch.float16)
out = flash_attn_func(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(out).all(), "FlashAttention produced nonfinite values"
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "cuda_extension_smoke": "passed", "sprig_inference": "not_yet_run"}, indent=2))
