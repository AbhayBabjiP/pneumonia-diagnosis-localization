"""PyTorch device selection with CUDA, Apple MPS, and CPU fallback."""


def select_device(torch_module=None):
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")
