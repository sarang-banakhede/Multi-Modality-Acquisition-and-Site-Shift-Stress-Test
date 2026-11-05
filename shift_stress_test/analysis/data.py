import numpy as np
import pandas as pd
from scipy.stats import norm

from ..data.manifest import group_info, subgroup_info
from ..data.records import external_ref, image_id, internal_ref, set_names
from ..degrade import GENERIC, SPECIFIC
from .levels import included


def probit(x):
    return norm.ppf(np.clip(x, 0.001, 0.999))


def load_eval(layout, models, dataset, seeds, subset, ckpt="best"):
    frames = []
    for m in models:
        for s in seeds:
            folder = layout.eval_dir(m, dataset, s, ckpt, subset)
            frames += [pd.read_parquet(f) for f in sorted(folder.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def make_halves(groups, strata, n_splits, rng):
    out = []
    for _ in range(n_splits):
        side = np.zeros(len(groups), bool)
        for st in np.unique(strata):
            members = np.nonzero(strata == st)[0]
            gids = rng.permutation(np.unique(groups[members]))
            count = [0, 0]
            for g in gids:
                sel = members[groups[members] == g]
                k = int(count[0] > count[1])
                side[sel] = k == 0
                count[k] += len(sel)
        out.append(side)
    return out


class DatasetTensor:
    def __init__(self, layout, cfg, dataset, models, seeds, levels, subset="test"):
        self.dataset, self.models, self.seeds = dataset, list(models), list(seeds)
        self.levels = list(levels)
        inc = included(layout, cfg, dataset, subset)
        self.K = {g: v["K"] for g, v in inc.items() if v["K"] > 0}
        self.excluded = [g for g, v in inc.items() if v["K"] == 0]
        self.reach = {g: v["share"] for g, v in inc.items()}
        self.generic = [g for g in GENERIC if g in self.K]
        self.specific = [g for g in SPECIFIC[dataset] if g in self.K]
        self.degs = self.generic + self.specific
        self.conds = [("clean", 0)] + [(g, k) for g in self.degs for k in range(1, self.K[g] + 1)]
        self.cidx = {c: i for i, c in enumerate(self.conds)}
        ref = internal_ref(dataset, subset)
        names = set_names(layout, ref, cfg.data.get("limit_per_split"))
        self.names = [image_id(n) for n in names]
        self.groups, self.strata = group_info(layout, ref, names)
        self.subgroup = subgroup_info(layout, ref, names)
        self.df = load_eval(layout, models, dataset, seeds, subset)
        self._cache = {}
        d = self.df
        keys = pd.Series(list(zip(d["degradation"], d["level"]))).map(self.cidx)
        keep = keys.notna().to_numpy()
        self.df = d = d[keep].reset_index(drop=True)
        self._c = keys[keep].to_numpy().astype(int)
        self._m = d["model"].map({m: i for i, m in enumerate(self.models)}).to_numpy()
        self._s = d["seed"].map({s: i for i, s in enumerate(self.seeds)}).to_numpy()
        self._n = d["image_id"].map({n: i for i, n in enumerate(self.names)}).to_numpy()
        self.family = {m: f for m, f in zip(d["model"], d["family"])}

    @property
    def shape(self):
        return len(self.models), len(self.seeds), len(self.conds), len(self.names)

    def tensor(self, metric="dice"):
        if metric not in self._cache:
            arr = np.full(self.shape, np.nan, np.float32)
            arr[self._m, self._s, self._c, self._n] = self.df[metric].to_numpy(np.float32)
            self._cache[metric] = arr
        return self._cache[metric]

    def score(self, metric="dice"):
        t = self.tensor(metric)
        return 1.0 - np.clip(t / 316.78, 0, 1) if metric in ("hd95", "assd") else t

    def cond_ids(self, g, kmax=None):
        return [self.cidx[(g, k)] for k in range(1, (self.K[g] if kmax is None else min(kmax, self.K[g])) + 1)]

    def complete(self, g, kmax=None):
        d = self.tensor("dice")[0, 0]
        return ~np.isnan(d[self.cond_ids(g, kmax)]).any(axis=0) & ~np.isnan(d[0])

    def cells(self, kmax=None):
        return [(g, k) for g in self.degs for k in range(1, (self.K[g] if kmax is None else min(kmax, self.K[g])) + 1)]

    def ssim(self):
        arr = np.full((len(self.conds), len(self.names)), np.nan, np.float32)
        first = self.df.drop_duplicates(["degradation", "level", "image_id"])
        c = pd.Series(list(zip(first["degradation"], first["level"]))).map(self.cidx).to_numpy().astype(int)
        n = first["image_id"].map({n: i for i, n in enumerate(self.names)}).to_numpy()
        arr[c, n] = first["achieved_ssim"].to_numpy(np.float32)
        return arr


class ExternalTensor:
    def __init__(self, layout, cfg, dataset, models, seeds):
        ref = external_ref(cfg, dataset)
        names = set_names(layout, ref, cfg.data.get("limit_per_split"))
        self.names = [image_id(n) for n in names]
        self.groups, _ = group_info(layout, ref, names)
        self.sources = np.array([n.split("_")[0] if ref.name == "cxr_mont_shen" else ref.name for n in self.names])
        df = load_eval(layout, models, dataset, seeds, "external")
        self.models, self.seeds = list(models), list(seeds)
        self.dice = np.full((len(models), len(seeds), len(self.names)), np.nan, np.float32)
        m = df["model"].map({x: i for i, x in enumerate(models)}).to_numpy()
        s = df["seed"].map({x: i for i, x in enumerate(seeds)}).to_numpy()
        n = df["image_id"].map({x: i for i, x in enumerate(self.names)}).to_numpy()
        self.dice[m, s, n] = df["dice"].to_numpy(np.float32)
