import torch
import torch.nn.functional as F


def seg_loss(logits, y):
    logits = logits.float()
    ce = F.cross_entropy(logits, y)
    p = logits.softmax(dim=1)[:, 1]
    g = (y == 1).float()
    inter = (p * g).flatten(1).sum(1)
    den = p.flatten(1).sum(1) + g.flatten(1).sum(1)
    dice = 1.0 - ((2.0 * inter + 1e-5) / (den + 1e-5)).mean()
    return dice + ce, dice, ce


def hard_dice(pred, y):
    p = (pred == 1).flatten(1).float()
    g = (y == 1).flatten(1).float()
    inter = (p * g).sum(1)
    den = p.sum(1) + g.sum(1)
    return torch.where(den > 0, 2.0 * inter / den.clamp(min=1.0), torch.ones_like(den))
