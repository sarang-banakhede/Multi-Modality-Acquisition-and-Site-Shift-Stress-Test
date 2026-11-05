import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from ..config import Config, seed_map
from ..layout import Layout
from ..metrics import FEATURE_COLUMNS, METRIC_COLUMNS, MetricPool
from ..models import model_info
from ..utils.io import load_json, save_npz, save_parquet
from ..utils.jobs import JobRegistry
from ..utils.log import get_logger, progress, setup_logging
from ..utils.memory import free_memory, resolve_device
from .engine import load_model, predict
from .units import DatasetData, phase_units


def job_key(phase, model, dataset, seed):
    return f"{phase}/{model}/{dataset}/seed{seed}"


def plan_jobs(phase, models, datasets, smap):
    every = sorted({s for m in models for s in smap[m]})
    return [job_key(phase, m, d, s) for d in datasets for s in every for m in models if s in smap[m]]


def levels_used(layout, cfg):
    path = layout.out / "degrade_config.json"
    return load_json(path)["levels"] if path.exists() else list(cfg.degrade.levels)


def unit_done(layout, model, dataset, seed, unit):
    return layout.eval_file(model, dataset, seed, unit.ckpt, unit.subset, unit.degradation).exists()


def job_done(layout, cfg, phase, model, dataset, seed):
    return all(unit_done(layout, model, dataset, seed, u) for u in phase_units(phase, dataset, cfg))


def evaluate_unit(layout, cfg, data, model, mean, std, precision, meta, unit, pool, device):
    mat = data.materialise(unit)
    preds, feats = predict(model, mean, std, mat["images"], cfg.infer.batch_size, device, precision)
    rows = np.array(pool.map(preds, mat["masks"], cfg.metrics.nsd_tolerance_px), dtype=np.float64)
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    for k in FEATURE_COLUMNS:
        df[k] = feats[k]
    head = {"model": meta["model"], "family": meta["family"], "dataset": meta["dataset"], "seed": meta["seed"],
            "ckpt": unit.ckpt, "subset": unit.subset, "degradation": unit.degradation}
    for k, v in head.items():
        df.insert(list(head).index(k), k, v)
    for pos, k in enumerate(["image_id", "level", "target_ssim", "achieved_ssim", "strength", "physical"]):
        df.insert(len(head) + pos, k, mat[k])
    d, m, s = meta["dataset"], meta["model"], meta["seed"]
    save_npz(layout.eval_masks(m, d, s, unit.ckpt, unit.subset, unit.degradation),
             masks=np.packbits(preds.reshape(len(preds), -1), axis=1), index=mat["index"], level=mat["level"])
    save_parquet(df, layout.eval_file(m, d, s, unit.ckpt, unit.subset, unit.degradation))
    del mat, preds, feats, df
    return None


def run_job(cfg, layout, phase, model_name, dataset, seed, data, pool, device, position=0):
    units = [u for u in phase_units(phase, dataset, cfg) if not unit_done(layout, model_name, dataset, seed, u)]
    if not units:
        return
    loaded = {}
    bar = progress(units, desc=f"{phase} {model_name}/{dataset}/s{seed}", position=position)
    info = model_info(model_name)
    meta = {"model": model_name, "family": info["family"], "dataset": dataset, "seed": seed}
    for unit in bar:
        bar.set_postfix(unit=f"{unit.ckpt}/{unit.subset}/{unit.degradation}")
        if unit.ckpt not in loaded:
            loaded.clear()
            free_memory(device)
            path = layout.run_dir(model_name, dataset, seed) / f"{unit.ckpt}.pt"
            loaded[unit.ckpt] = load_model(path, device, cfg.data.num_classes, cfg.train.channels_last)
        model, mean, std, precision = loaded[unit.ckpt]
        evaluate_unit(layout, cfg, data, model, mean, std, precision, meta, unit, pool, device)
        free_memory(device)
    loaded.clear()
    free_memory(device)


def _worker(cfg_dict, phase, rank, dev, keys):
    cfg = Config(cfg_dict)
    layout = Layout(cfg)
    setup_logging(layout.log_file, tag=f"{phase}-{dev}{rank}")
    pool = MetricPool(cfg.runtime.metric_workers)
    device = resolve_device("cpu" if dev == "cpu" else cfg.runtime.device, 0 if dev == "cpu" else dev)
    reg = JobRegistry(layout.jobs_file, layout.events_file)
    levels = levels_used(layout, cfg)
    data, current = None, None
    name = f"{phase}-{dev}{rank}"
    while True:
        key = reg.claim(keys, name)
        if key is None:
            break
        _, model_name, dataset, seed = key.split("/")
        try:
            if current != dataset:
                if data is not None:
                    data.release()
                data, current = DatasetData(layout, cfg, dataset, levels), dataset
            reg.update(key, status="running", worker=name)
            run_job(cfg, layout, phase, model_name, dataset, int(seed[4:]), data, pool, device, rank)
            reg.update(key, status="done")
        except Exception as e:
            get_logger().info(f"FAILED {key}: {e}\n{traceback.format_exc()}")
            reg.update(key, status="failed", error=str(e)[:500])
            reg.event(name, key, "failed", str(e)[:200])
        free_memory(device)
    pool.close()


def run_inference(cfg, layout, phase, models, datasets, seeds):
    log = get_logger()
    if phase == "infer_test" and not (layout.prereg_dir / "prereg.json").exists():
        raise SystemExit("prereg.json is missing: run the 'freeze' stage before touching the test set")
    keys = plan_jobs(phase, models, datasets, seed_map(cfg, models, seeds))
    reg = JobRegistry(layout.jobs_file, layout.events_file)
    reg.prepare({k: job_done(layout, cfg, *_parts(k)) for k in keys})
    state = reg.read()
    pending = [k for k in keys if state[k]["status"] != "done"]
    log.info(f"{phase}: {len(keys) - len(pending)} done, {len(pending)} to run")
    if not pending:
        return
    if cfg.runtime.device == "cpu" or not torch.cuda.is_available():
        gpus = ["cpu"] * max(1, int(cfg.runtime.cpu_workers))
    else:
        gpus = cfg.runtime.gpus if cfg.runtime.gpus is not None else list(range(torch.cuda.device_count()))
    if len(gpus) == 1:
        _worker(cfg.to_dict(), phase, 0, gpus[0], pending)
        setup_logging(layout.log_file, tag="main")
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(cfg.to_dict(), phase, i, g, pending)) for i, g in enumerate(gpus)]
        for p in procs:
            p.start()
            time.sleep(3)
        for p in procs:
            p.join()
    state = reg.read()
    failed = [k for k in pending if state[k]["status"] == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} {phase} job(s) failed, see the log: {failed[:3]}")


def _parts(key):
    phase, model, dataset, seed = key.split("/")
    return phase, model, dataset, int(seed[4:])
