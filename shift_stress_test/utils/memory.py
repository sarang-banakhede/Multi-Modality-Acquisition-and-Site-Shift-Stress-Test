import gc

import torch


def free_memory(device=None):
    gc.collect()
    if torch.cuda.is_available() and (device is None or torch.device(device).type == "cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def resolve_device(setting, index=0):
    if setting == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")
