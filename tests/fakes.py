import numpy as np

GENERIC = ["G1", "G2", "G3"]


class FakeTensor:
    def __init__(self, rng, models=8, seeds=3, n=60, specific=("D1", "D2", "D3"), k=4, relation=0.0, shared=0.0, noise=0.08):
        self.dataset = "fake"
        self.models = [f"m{i}" for i in range(models)]
        self.seeds = list(range(seeds))
        self.generic, self.specific = list(GENERIC), list(specific)
        self.degs = self.generic + self.specific
        self.K = {g: k for g in self.degs}
        self.names = [str(i) for i in range(n)]
        self.groups = np.arange(n) // 2
        self.strata = np.array(["a"] * n)
        self.family = {m: "f" for m in self.models}
        self.levels = [0.95 - 0.05 * i for i in range(k)]
        self.conds = [("clean", 0)] + [(g, j) for g in self.degs for j in range(1, k + 1)]
        self.cidx = {c: i for i, c in enumerate(self.conds)}
        skill = rng.normal(0, 0.1, models)
        self.rob = rng.normal(0, 0.1, models)
        data = np.zeros((models, seeds, len(self.conds), n), np.float32)
        image = rng.normal(0, 0.1, n)
        for ci, (g, j) in enumerate(self.conds):
            cond_skill = relation * skill + np.sqrt(max(1 - relation ** 2, 0)) * rng.normal(0, 0.1, models)
            offset = skill if g == "clean" else cond_skill + shared * self.rob
            base = 0.8 + image[None, None] + offset[:, None, None] - 0.03 * j
            data[:, :, ci] = np.clip(base + rng.normal(0, noise, (models, seeds, n)), 0.01, 0.99)
        self._t = data

    def tensor(self, metric="dice"):
        return self._t

    def score(self, metric="dice"):
        return self._t

    def cond_ids(self, g, kmax=None):
        return [self.cidx[(g, j)] for j in range(1, (self.K[g] if kmax is None else min(kmax, self.K[g])) + 1)]

    def complete(self, g, kmax=None):
        return np.ones(len(self.names), bool)


class FakeExternal:
    def __init__(self, rng, T, shared=0.0):
        n = 40
        self.names = [str(i) for i in range(n)]
        self.groups = np.arange(n)
        self.sources = np.array(["ext"] * n)
        base = 0.75 + (shared * T.rob + rng.normal(0, 0.1, len(T.rob)))[:, None, None]
        self.dice = np.clip(base + rng.normal(0, 0.08, (len(T.rob), len(T.seeds), n)), 0.01, 0.99).astype(np.float32)

