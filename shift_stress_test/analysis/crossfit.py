import numpy as np
from scipy.stats import rankdata

from .data import make_halves
from .meta import fisher, spearman_rows
from .scores import LINKS, area, curves, er_curves


class Clusters:
    def __init__(self, idx, groups):
        self.idx = np.asarray(idx)
        if len(self.idx) == 0:
            self.members = []
            return
        _, inv = np.unique(groups[self.idx], return_inverse=True)
        self.members = [self.idx[inv == i] for i in range(inv.max() + 1)]

    def draw(self, rng):
        if not self.members:
            return self.idx
        pick = rng.integers(0, len(self.members), len(self.members))
        return np.concatenate([self.members[i] for i in pick])


def seed_weights(rng, m, s):
    return rng.multinomial(s, np.full(s, 1.0 / s), size=m) / s


def standardise(v):
    return (v - v.mean(-1, keepdims=True)) / (v.std(-1, keepdims=True) + 1e-9)


class DatasetContext:
    def __init__(self, T, ext, n_splits, rng, metric="dice", link="probit", kmax=None, crossfit=True, source=None):
        self.T, self.ext, self.link_name, self.link = T, ext, link, LINKS[link]
        self.crossfit = crossfit
        self.M, self.S = len(T.models), len(T.seeds)
        self.groups = T.groups
        self.halves = make_halves(T.groups, T.strata, n_splits if crossfit else 1, rng)
        score = T.score(metric)
        self.generic, self.specific, self.degs = T.generic, T.specific, T.degs
        self.K = {g: min(kmax, T.K[g]) if kmax else T.K[g] for g in T.degs}
        self.cells = [(g, k) for g in T.degs for k in range(1, self.K[g] + 1)]
        self.tg = {g: score[:, :, [0] + T.cond_ids(g, kmax)] for g in T.degs}
        self.t0 = score[:, :, [0]]
        self.complete = {g: T.complete(g, kmax) for g in T.degs}
        self.clusters = {}
        self.ext_mask = np.ones(len(ext.names), bool) if ext is None or source is None else ext.sources == source
        self.ext_clusters = (Clusters(np.nonzero(self.ext_mask)[0], ext.groups)
                             if ext is not None and self.ext_mask.any() else None)

    def half_idx(self, split, side, g=None):
        if not self.crossfit:
            mask = np.ones(len(self.groups), bool)
        else:
            mask = self.halves[split] if side == 0 else ~self.halves[split]
        if g is not None:
            mask = mask & self.complete[g]
        return np.nonzero(mask)[0]

    def half_clusters(self, split, side, g=None):
        key = (split, side, g)
        if key not in self.clusters:
            self.clusters[key] = Clusters(self.half_idx(split, side, g), self.groups)
        return self.clusters[key]

    def curve(self, g, idx, w=None):
        return curves(self.tg[g], list(range(self.tg[g].shape[2])), idx, w)

    def clean(self, idx, w=None):
        return curves(self.t0, [0], idx, w)[:, 0]

    def auc_er(self, gs, idx_of, w=None):
        vals = []
        for g in gs:
            idx = idx_of(g)
            if len(idx) == 0:
                return None
            vals.append(area(er_curves(self.curve(g, idx, w), self.link)))
        return np.mean(vals, axis=0) if vals else None

    def ext_dice(self, idx=None, w=None):
        d = self.ext.dice[..., self.ext_mask] if idx is None else self.ext.dice[..., idx]
        per_seed = np.nanmean(d, axis=-1)
        return per_seed.mean(1) if w is None else (per_seed * w).sum(1)


def h1_arrays(ctx, split, side_x, w=None, boot_rng=None, rank=False):
    zx, zy = [], []
    for g in ctx.degs:
        cx, cy = ctx.half_clusters(split, side_x, g), ctx.half_clusters(split, 1 - side_x, g)
        ix = cx.draw(boot_rng) if boot_rng is not None else cx.idx
        iy = cy.draw(boot_rng) if boot_rng is not None else cy.idx
        if len(ix) == 0 or len(iy) == 0:
            return None
        x = ctx.link(ctx.clean(ix, w))
        y = ctx.link(ctx.curve(g, iy, w)[:, 1:]).T
        zx.append(np.repeat(x[None], len(y), axis=0))
        zy.append(y)
    if not zx:
        return None
    zx, zy = np.concatenate(zx), np.concatenate(zy)
    if rank:
        zx, zy = rankdata(zx, axis=-1), rankdata(zy, axis=-1)
    return standardise(zx), standardise(zy)


def cell_corr(zx, zy):
    return (zx * zy).mean(-1)


def h1_point(ctx, rank=False):
    pairs = []
    for s in range(len(ctx.halves)):
        for side in (0, 1):
            a = h1_arrays(ctx, s, side, rank=rank)
            if a is not None:
                pairs.append(a)
    return pairs


def h1_perm(pairs, perms):
    out = np.zeros(len(perms))
    for zx, zy in pairs:
        out += np.einsum("cpm,cm->pc", zx[:, perms], zy).mean(1) / zx.shape[-1]
    return out / len(pairs)


def h1_boot(ctx, reps, rng):
    out = []
    for _ in range(reps):
        s = int(rng.integers(len(ctx.halves)))
        w = seed_weights(rng, ctx.M, ctx.S)
        vals = []
        for side in (0, 1):
            a = h1_arrays(ctx, s, side, w, rng)
            if a is not None:
                vals.append(cell_corr(*a).mean())
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def _picker(ctx, s, side, rng):
    def f(g):
        c = ctx.half_clusters(s, side, g)
        return c.draw(rng) if rng is not None else c.idx
    return f


def rho_pairs_h2(ctx, split=None, rng=None):
    out = []
    if not ctx.generic or not ctx.specific:
        return out
    for s in (range(len(ctx.halves)) if split is None else [split]):
        for side in (0, 1):
            w = seed_weights(rng, ctx.M, ctx.S) if rng is not None else None
            a = ctx.auc_er(ctx.generic, _picker(ctx, s, side, rng), w)
            b = ctx.auc_er(ctx.specific, _picker(ctx, s, 1 - side, rng), w)
            if a is not None and b is not None:
                out.append((a, b))
    return out


def rho_pairs_h4(ctx, split=None, rng=None):
    out = []
    if ctx.ext is None or ctx.ext_clusters is None:
        return out
    for s in (range(len(ctx.halves)) if split is None else [split]):
        for side in (0, 1):
            w = seed_weights(rng, ctx.M, ctx.S) if rng is not None else None
            a = ctx.auc_er(ctx.degs, _picker(ctx, s, side, rng), w)
            other = ctx.half_clusters(s, 1 - side)
            idx_b = other.draw(rng) if rng is not None else other.idx
            ext_idx = ctx.ext_clusters.draw(rng) if rng is not None else None
            if a is None or len(idx_b) == 0:
                continue
            d = np.stack([ctx.clean(idx_b, w), ctx.ext_dice(ext_idx, w)], axis=-1)
            out.append((a, er_curves(d, ctx.link)[:, 1]))
    return out


def mean_rho(pairs):
    return float(np.mean([spearman_rows(a, b) for a, b in pairs])) if pairs else np.nan


def pair_rank_perm(pairs, perms):
    ra = np.array([rankdata(a) for a, _ in pairs])
    rb = np.array([rankdata(b) for _, b in pairs])
    ra, rb = ra - ra.mean(-1, keepdims=True), rb - rb.mean(-1, keepdims=True)
    rbp = rb[:, perms]
    num = np.einsum("sm,spm->sp", ra, rbp)
    den = np.sqrt((ra ** 2).sum(-1)[:, None] * (rbp ** 2).sum(-1))
    return (num / (den + 1e-12)).mean(0)


def boot_rho(fn, ctx, reps, rng):
    out = []
    for _ in range(reps):
        s = int(rng.integers(len(ctx.halves)))
        out.append(mean_rho(fn(ctx, s, rng)))
    return np.array(out)


def fisher_mean(rhos):
    return float(np.mean(fisher(np.array(rhos))))
