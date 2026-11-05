import numpy as np
import torch

from ..data.cache import ensure_norm, load_cache
from ..data.records import internal_ref
from ..utils.seed import stable_seed

TRAINING_SPLITS = ("train", "val")


def load_training_split(layout, cfg, dataset, split):
    assert split in TRAINING_SPLITS, "training never reads the test split or an external set"
    return load_cache(layout, cfg, internal_ref(dataset, split))


class GPUData:
    def __init__(self, images, masks, device):
        self.x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(masks).to(device)
        self.n = self.x.shape[0]

    def subset(self, idx):
        idx = torch.as_tensor(idx, device=self.x.device)
        out = object.__new__(GPUData)
        out.x, out.y, out.n = self.x[idx], self.y[idx], len(idx)
        return out


def norm_tensors(layout, cfg, dataset, device):
    norm = ensure_norm(layout, cfg, dataset)
    mean = torch.tensor(norm["mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(norm["std"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return norm, mean, std


def get_batch(data, idx, mean, std, aug_cfg=None, flags=None, gen=None):
    from .augment import augment

    idx = idx.to(data.x.device)
    x = data.x[idx].float().div_(255.0)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1).contiguous()
    y = data.y[idx].long()
    if aug_cfg is not None:
        x, y = augment(x, y, aug_cfg, flags, gen)
    return (x - mean) / std, y


def trainsub_indices(dataset, n, size):
    rng = np.random.default_rng(stable_seed(dataset, "trainsub"))
    return np.sort(rng.choice(n, min(size, n), replace=False))


class Sampler:
    def __init__(self, n, seed):
        self.n = n
        self.g = torch.Generator()
        self.g.manual_seed(seed)
        self.perm = torch.randperm(n, generator=self.g)
        self.pos = 0

    def next(self, b):
        if self.n < b:
            return torch.randint(self.n, (b,), generator=self.g)
        if self.pos + b > self.n:
            self.perm = torch.randperm(self.n, generator=self.g)
            self.pos = 0
        idx = self.perm[self.pos:self.pos + b]
        self.pos += b
        return idx

    def state(self):
        return {"g": self.g.get_state(), "perm": self.perm.clone(), "pos": self.pos}

    def load(self, s):
        self.g.set_state(s["g"].cpu())
        self.perm = s["perm"].cpu().clone()
        self.pos = s["pos"]
