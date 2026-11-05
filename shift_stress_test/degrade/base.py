import numpy as np

from ..utils.seed import stable_seed

GENERIC = ["G1", "G2", "G3"]
SPECIFIC = {"isic2016": ["D1", "D2", "D3"], "kvasir_seg": ["E1", "E2", "E3"], "tn3k": ["U1", "U2", "U3"],
            "refuge2": ["F1", "F2", "F3"], "darwin_cxr": ["X1", "X2", "X3"]}


class Degradation:
    name = ""
    unit = ""

    def __init__(self, params):
        self.params = params

    def rng(self, dataset, image_id):
        return np.random.default_rng(stable_seed(dataset, image_id, self.name))

    def prepare(self, img, rng):
        return None

    def apply(self, img, state, s):
        raise NotImplementedError

    def physical(self, s):
        return float(s) * float(self.params.get("max", 1.0))

    def __call__(self, img, state, s):
        if s <= 0:
            return img.copy()
        return np.clip(self.apply(img, state, float(s)), 0, 255).astype(np.uint8)


def degradations_for(dataset):
    return GENERIC + SPECIFIC[dataset]


def build_degradation(name, cfg):
    from . import cxr, dermoscopy, endoscopy, fundus, generic, ultrasound

    registry = {}
    for mod in (generic, dermoscopy, endoscopy, ultrasound, fundus, cxr):
        registry.update(mod.REGISTRY)
    return registry[name](cfg.degrade.params[name].to_dict())


def to_float(img):
    return img.astype(np.float32)


def to_linear(img):
    return (img.astype(np.float32) / 255.0) ** 2.2


def from_linear(x):
    return np.clip(x, 0.0, 1.0) ** (1 / 2.2) * 255.0


def short_side(img):
    return min(img.shape[:2])
