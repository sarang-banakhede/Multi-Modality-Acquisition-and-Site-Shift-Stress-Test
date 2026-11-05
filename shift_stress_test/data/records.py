from collections import namedtuple
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg"}

SetRef = namedtuple("SetRef", ["dataset", "name", "split"])


def internal_ref(dataset, split):
    return SetRef(dataset, dataset, split)


def external_ref(cfg, dataset):
    return SetRef(dataset, cfg.experiment.external[dataset], None)


def subset_ref(cfg, dataset, subset):
    if subset == "external":
        return external_ref(cfg, dataset)
    return internal_ref(dataset, subset)


def set_dir(layout, ref):
    base = layout.data / ref.name
    return base if ref.split is None else base / ref.split


def list_set(layout, ref, limit=None):
    img_dir = set_dir(layout, ref) / "images"
    msk_dir = set_dir(layout, ref) / "masks"
    names = sorted(p.name for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not names:
        raise FileNotFoundError(f"no images in {img_dir}")
    if limit:
        names = names[:limit]
    for n in names:
        if not (msk_dir / n).exists():
            raise FileNotFoundError(f"missing mask {msk_dir / n}")
    return names, [img_dir / n for n in names], [msk_dir / n for n in names]


def set_names(layout, ref, limit=None):
    if (set_dir(layout, ref) / "images").is_dir():
        return list_set(layout, ref, limit)[0]
    from ..utils.io import load_json
    items = load_json(layout.split_manifest(ref.name))["items"]
    if ref.split is not None:
        items = [it for it in items if it.get("split") == ref.split]
    names = sorted(Path(it["image"]).name if "image" in it else f"{it['id']}.png" for it in items)
    return names[:limit] if limit else names


def image_id(name):
    return name.rsplit(".", 1)[0]
