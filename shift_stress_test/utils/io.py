import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def _tmp(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + f".tmp{os.getpid()}")


def save_json(obj, path):
    tmp = _tmp(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_torch(obj, path):
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_npz(path, **arrays):
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def save_parquet(df, path):
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def truncate_csv(path, max_iteration):
    path = Path(path)
    if not path.exists():
        return
    with open(path, errors="replace") as f:
        lines = f.read().replace("\x00", "").splitlines()
    rows = [r for r in csv.DictReader(lines) if None not in r.values() and None not in r]
    keep = []
    for r in rows:
        try:
            if int(r["iteration"]) <= max_iteration:
                keep.append(r)
        except (TypeError, ValueError):
            pass
    if not keep:
        path.unlink()
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()
