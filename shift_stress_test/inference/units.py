from dataclasses import dataclass

import numpy as np

from ..data.cache import load_cache
from ..data.records import image_id, subset_ref
from ..degrade import GENERIC, SPECIFIC


@dataclass(frozen=True)
class Unit:
    ckpt: str
    subset: str
    degradation: str


def phase_units(phase, dataset, cfg):
    last_sets = set(cfg.infer.last_checkpoint_sets)
    if phase == "infer_val":
        return [Unit("best", "val", d) for d in ["clean"] + GENERIC]
    units = [Unit("best", "test", d) for d in ["clean"] + GENERIC + SPECIFIC[dataset]]
    units.append(Unit("best", "external", "clean"))
    units += [Unit("last", s, "clean") for s in ("test", "external") if s in last_sets]
    return units


class DatasetData:
    def __init__(self, layout, cfg, dataset, levels):
        self.layout, self.cfg, self.dataset, self.levels = layout, cfg, dataset, levels
        self.clean, self.degraded = {}, {}

    def get_clean(self, subset):
        if subset not in self.clean:
            self.clean[subset] = load_cache(self.layout, self.cfg, subset_ref(self.cfg, self.dataset, subset))
        return self.clean[subset]

    def get_degraded(self, subset, name):
        key = (subset, name)
        if key not in self.degraded:
            with np.load(self.layout.degraded_file(self.dataset, subset, name)) as z:
                self.degraded[key] = {k: z[k] for k in z.files}
        return self.degraded[key]

    def materialise(self, unit):
        clean = self.get_clean(unit.subset)
        names = clean["names"]
        if unit.degradation == "clean":
            n = len(names)
            return {"images": clean["images"], "masks": clean["masks"], "image_id": [image_id(n) for n in names],
                    "level": np.zeros(n, int), "target_ssim": np.full(n, np.nan), "achieved_ssim": np.ones(n),
                    "strength": np.zeros(n), "physical": np.zeros(n), "index": np.arange(n)}
        deg = self.get_degraded(unit.subset, unit.degradation)
        assert [str(n) for n in deg["names"]] == names, "degraded set and cache disagree on image order"
        levels = len(self.levels)
        reach = deg["reachable"][:levels]
        lv, idx = np.nonzero(reach)
        return {"images": deg["images"][lv, idx], "masks": clean["masks"][idx], "image_id": [image_id(names[i]) for i in idx],
                "level": lv + 1, "target_ssim": np.array(self.levels)[lv], "achieved_ssim": deg["ssim"][lv, idx],
                "strength": deg["strength"][lv, idx], "physical": deg["physical"][lv, idx], "index": idx}

    def release(self):
        self.clean.clear()
        self.degraded.clear()
