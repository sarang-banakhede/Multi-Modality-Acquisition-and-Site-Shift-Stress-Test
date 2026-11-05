import contextlib

import torch


def resolve_precision(mode, device):
    if device.type != "cuda":
        return "bf16" if mode == "bf16" else "fp32"
    if mode != "auto":
        return mode
    if "GTX 16" in torch.cuda.get_device_name(device):
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def autocast_context(precision, device):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(torch.device(device).type, dtype=dtype)


def make_scaler(precision, device):
    return torch.amp.GradScaler("cuda", enabled=precision == "fp16" and torch.device(device).type == "cuda")


def enable_fast_math():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def prepare_model(model, enabled, device):
    if enabled and torch.device(device).type == "cuda":
        model.to(memory_format=torch.channels_last)
        model._channels_last = True
    return model


def prep_input(x, model):
    return x.contiguous(memory_format=torch.channels_last) if getattr(model, "_channels_last", False) else x
