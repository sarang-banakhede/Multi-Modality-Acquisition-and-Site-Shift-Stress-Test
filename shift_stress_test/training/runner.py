import time
import traceback

import torch
import torch.multiprocessing as mp

from ..config import Config, seed_map
from ..data.cache import build_cache, ensure_norm
from ..data.records import internal_ref
from ..layout import Layout
from ..metrics import MetricPool
from ..utils.jobs import JobRegistry
from ..utils.log import get_logger, setup_logging
from ..utils.memory import free_memory, resolve_device
from ..utils.precision import enable_fast_math
from .trainer import StopFlag, Trainer


def job_key(model, dataset, seed):
    return f"train/{model}/{dataset}/seed{seed}"


def plan_jobs(models, datasets, smap):
    every = sorted({s for m in models for s in smap[m]})
    return [(m, d, s) for s in every for d in datasets for m in models if s in smap[m]]


def _worker(cfg_dict, rank, dev, keys, deadline):
    cfg = Config(cfg_dict)
    layout = Layout(cfg)
    setup_logging(layout.log_file, tag=f"train-{dev}{rank}")
    pool = MetricPool(cfg.runtime.metric_workers)
    stop = StopFlag()
    stop.install()
    device = resolve_device("cpu" if dev == "cpu" else cfg.runtime.device, 0 if dev == "cpu" else dev)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        enable_fast_math()
    reg = JobRegistry(layout.jobs_file, layout.events_file)
    name = f"{dev}{rank}"
    failed = 0
    while not stop.requested:
        key = reg.claim(keys, name)
        if key is None:
            break
        _, model, dataset, seed = key.split("/")
        try:
            status = Trainer(cfg, layout, model, dataset, int(seed[4:]), device, pool, stop, reg, name, deadline,
                             position=rank).run()
        except Exception as e:
            failed += 1
            get_logger().info(f"FAILED {key}: {e}\n{traceback.format_exc()}")
            reg.update(key, status="failed", error=str(e)[:500])
            reg.event(name, key, "failed", str(e)[:200])
            status = "failed"
        free_memory(device)
        if status == "paused":
            break
    pool.close()
    return failed


def run_training(cfg, layout, models, datasets, seeds):
    log = get_logger()
    keys_all = [job_key(*j) for j in plan_jobs(models, datasets, seed_map(cfg, models, seeds))]
    reg = JobRegistry(layout.jobs_file, layout.events_file)
    reg.prepare({k: (layout.run_dir(*_parts(k)) / "DONE").exists() for k in keys_all})
    state = reg.read()
    pending = [k for k in keys_all if state[k]["status"] != "done"]
    log.info(f"training: {len(keys_all) - len(pending)} done, {len(pending)} to run")
    if not pending:
        return
    for d in datasets:
        for split in ("train", "val"):
            build_cache(layout, cfg, internal_ref(d, split))
        ensure_norm(layout, cfg, d)
    if cfg.runtime.device == "cpu" or not torch.cuda.is_available():
        gpus = ["cpu"] * max(1, int(cfg.runtime.cpu_workers))
    else:
        gpus = cfg.runtime.gpus if cfg.runtime.gpus is not None else list(range(torch.cuda.device_count()))
    limit = cfg.runtime.time_limit_hours
    deadline = time.time() + limit * 3600.0 if limit else None
    if len(gpus) == 1:
        _worker(cfg.to_dict(), 0, gpus[0], pending, deadline)
        setup_logging(layout.log_file, tag="main")
    else:
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(cfg.to_dict(), i, g, pending, deadline)) for i, g in enumerate(gpus)]
        for p in procs:
            p.start()
            time.sleep(3)
        for p in procs:
            p.join()
    state = reg.read()
    left = [k for k in pending if state[k]["status"] != "done"]
    failed = [k for k in left if state[k]["status"] == "failed"]
    if failed:
        raise RuntimeError(f"{len(failed)} training job(s) failed, see the log: {failed[:3]}")
    if left:
        raise SystemExit(f"training paused with {len(left)} job(s) left; rerun the same command to resume")


def _parts(key):
    _, model, dataset, seed = key.split("/")
    return model, dataset, int(seed[4:])
