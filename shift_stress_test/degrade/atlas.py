import numpy as np
from PIL import Image, ImageDraw

from ..data.cache import load_cache
from ..data.records import internal_ref
from .base import degradations_for


def _tile(img, size=112):
    img = img if img.shape[-1] == 3 else np.repeat(img, 3, axis=-1)
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def make_atlas(cfg, layout, dataset, n_images=3, split="val"):
    path = layout.degraded_dir(dataset, split) / "atlas.png"
    clean = load_cache(layout, cfg, internal_ref(dataset, split))
    picks = np.linspace(0, len(clean["names"]) - 1, min(n_images, len(clean["names"]))).astype(int)
    blocks = []
    for deg in degradations_for(dataset):
        f = layout.degraded_file(dataset, split, deg)
        if not f.exists():
            continue
        with np.load(f) as z:
            images, reach, levels = z["images"], z["reachable"], len(z["levels"])
        cols = sorted({1, max(levels // 2, 1), levels})
        rows = []
        for i in picks:
            tiles = [_tile(clean["images"][i])]
            tiles += [_tile(images[k - 1, i]) if reach[k - 1, i] else np.zeros((112, 112, 3), np.uint8) for k in cols]
            rows.append(np.concatenate(tiles, axis=1))
        block = Image.fromarray(np.concatenate(rows, axis=0))
        ImageDraw.Draw(block).text((3, 3), deg, fill=(255, 255, 0))
        blocks += [np.asarray(block), np.full((6, block.width, 3), 255, np.uint8)]
    if blocks:
        Image.fromarray(np.concatenate(blocks, axis=0)).save(path)
    return path
