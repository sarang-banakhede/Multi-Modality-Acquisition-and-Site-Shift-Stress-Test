from pathlib import Path

import yaml


class Config:
    def __init__(self, data):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, key):
        try:
            value = self._d[key]
        except KeyError:
            raise AttributeError(key)
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key):
        value = self._d[key]
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key):
        return key in self._d

    def get(self, key, default=None):
        return self[key] if key in self._d else default

    def keys(self):
        return self._d.keys()

    def items(self):
        return ((k, self[k]) for k in self._d)

    def to_dict(self):
        return self._d


def _set(tree, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        tree = tree.setdefault(k, {})
    tree[keys[-1]] = value


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _read(path):
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    parent = data.pop("base", None)
    if parent:
        return _merge(_read(path.parent / parent), data)
    return data


def load_config(path, overrides=()):
    data = _read(path)
    for item in overrides:
        key, _, raw = item.partition("=")
        _set(data, key.strip(), yaml.safe_load(raw))
    return Config(data)


def seed_map(cfg, models, seeds):
    pt = list(cfg.experiment.get("pretrained_seeds", seeds))
    return {m: (pt if m.endswith("_pt") else list(seeds)) for m in models}


def resolve(cfg):
    out = Path(cfg.paths.output).expanduser().resolve()
    data = Path(cfg.paths.data).expanduser().resolve()
    cache = cfg.paths.get("cache")
    cache = Path(cache).expanduser().resolve() if cache else out / "cache"
    splits = cfg.paths.get("splits")
    splits = Path(splits).expanduser().resolve() if splits else data / "splits"
    return data, out, cache, splits
