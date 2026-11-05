import numpy as np
import torch

from ..utils.precision import autocast_context, prep_input, prepare_model, resolve_precision
from ..metrics import FEATURE_COLUMNS, confidence_features
from ..models import build_model


def load_model(path, device, num_classes=2, channels_last=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], num_classes, ckpt["img_size"])
    model.load_state_dict(ckpt["model"])
    model = prepare_model(model.to(device), channels_last, device).eval()
    norm = ckpt["norm"]
    mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(norm["std"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    precision = resolve_precision(ckpt.get("precision", "fp32"), device)
    return model, mean, std, precision


@torch.no_grad()
def predict(model, mean, std, images, batch_size, device, precision="fp32"):
    preds, feats = [], {k: [] for k in FEATURE_COLUMNS}
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        with autocast_context(precision, device):
            logits = model(prep_input((x - mean) / std, model))
        pred, f = confidence_features(logits)
        preds.append(pred.cpu().numpy())
        for k in FEATURE_COLUMNS:
            feats[k].append(f[k])
        del x, pred
    return np.concatenate(preds), {k: np.concatenate(v) for k, v in feats.items()}
