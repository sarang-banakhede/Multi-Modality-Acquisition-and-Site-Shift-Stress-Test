from pathlib import Path

from .config import resolve


class Layout:
    def __init__(self, cfg):
        self.data, self.out, self.cache, self.splits = resolve(cfg)

    @property
    def log_file(self):
        return self.out / "logs" / "pipeline.log"

    @property
    def jobs_file(self):
        return self.out / "state" / "jobs.json"

    @property
    def events_file(self):
        return self.out / "state" / "events.csv"

    @property
    def gates_file(self):
        return self.out / "state" / "gates.json"

    def split_manifest(self, name):
        return self.splits / f"{name}.json"

    def cache_file(self, name, split):
        return self.cache / f"{name}_{split}.npz"

    def norm_file(self, dataset):
        return self.cache / "norm" / f"{dataset}.json"

    def run_dir(self, model, dataset, seed):
        return self.out / "train" / model / dataset / f"seed{seed}"

    def degraded_dir(self, dataset, split):
        return self.out / "degraded" / dataset / split

    def degraded_file(self, dataset, split, degradation):
        return self.degraded_dir(dataset, split) / f"{degradation}.npz"

    def degrade_report(self, dataset, split):
        return self.degraded_dir(dataset, split) / "reachability.csv"

    def eval_dir(self, model, dataset, seed, ckpt, subset):
        return self.out / "eval" / model / dataset / f"seed{seed}" / ckpt / subset

    def eval_file(self, model, dataset, seed, ckpt, subset, degradation):
        return self.eval_dir(model, dataset, seed, ckpt, subset) / f"{degradation}.parquet"

    def eval_masks(self, model, dataset, seed, ckpt, subset, degradation):
        return self.eval_dir(model, dataset, seed, ckpt, subset) / f"{degradation}.masks.npz"

    @property
    def prereg_dir(self):
        return self.out / "prereg"

    @property
    def analysis_dir(self):
        return self.out / "analysis"

    def ensure(self):
        for p in (self.out / "logs", self.out / "state", self.cache / "norm"):
            Path(p).mkdir(parents=True, exist_ok=True)
