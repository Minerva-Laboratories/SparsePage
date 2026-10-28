"""JIT build of the CUDA kernels (csrc/sparsepage.cu), once per head_dim, cached under ~/.cache/sparsepage (SPARSEPAGE_BUILD_DIR)."""
import os

import torch

_EXT = {}


def ext(head_dim: int = 256):
    if head_dim not in _EXT:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:           # build for the GPU in this machine
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        name = f"sparsepage_kernels_d{head_dim}"
        bd = os.path.join(os.environ.get("SPARSEPAGE_BUILD_DIR", os.path.expanduser("~/.cache/sparsepage")), name)
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "sparsepage.cu")
        _EXT[head_dim] = load(name, [src], build_directory=bd, verbose=False,
                              extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler", f"-DRF_D={head_dim}"])
    return _EXT[head_dim]
