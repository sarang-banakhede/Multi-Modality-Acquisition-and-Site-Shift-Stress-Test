import numpy as np

from ..utils.io import load_json
from .records import image_id


def _union_find_groups(ids, pairs):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in pairs:
        a, b = str(a), str(b)
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in ids])


def group_info(layout, ref, names):
    manifest = load_json(layout.split_manifest(ref.name))
    ids = [image_id(n) for n in names]
    if ref.split is None:
        groups = _union_find_groups(ids, manifest.get("within_set_near_duplicate_pairs", []))
        return groups, np.array(["all"] * len(ids))
    items = {str(it["id"]): it for it in manifest["items"] if it.get("split") == ref.split}
    raw = [items[i]["group"] for i in ids]
    codes = {g: k for k, g in enumerate(dict.fromkeys(raw))}
    return np.array([codes[g] for g in raw]), np.array([items[i]["stratum"] for i in ids])


def subgroup_info(layout, ref, names):
    if ref.split is None:
        return None
    manifest = load_json(layout.split_manifest(ref.name))
    items = {str(it["id"]): it for it in manifest["items"] if it.get("split") == ref.split}
    groups = [items[image_id(n)].get("subgroup") for n in names]
    return np.array(groups) if all(g is not None for g in groups) else None
