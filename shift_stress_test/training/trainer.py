import math
import platform
import random
import signal
import socket
import sys
import time

import numpy as np
import torch

from ..metrics.segmentation import METRIC_COLUMNS
from ..models import build_model, model_info
from ..utils.io import append_csv, save_json, save_torch, sha256_file, truncate_csv
from ..utils.jobs import now
from ..utils.log import get_logger, progress
from ..utils.memory import free_memory
from ..utils.precision import autocast_context, make_scaler, prep_input, prepare_model, resolve_precision
from ..utils.seed import seed_everything
from .data import GPUData, Sampler, get_batch, load_training_split, norm_tensors, trainsub_indices
from .losses import hard_dice, seg_loss

EVAL_KEYS = ["dice", "iou", "precision", "recall", "hd95", "assd", "nsd", "area_error"]


class StopFlag:
    def __init__(self):
        self.requested = False

    def install(self):
        def handler(signum, frame):
            if self.requested:
                raise KeyboardInterrupt
            self.requested = True
            get_logger().info(f"signal {signum}: saving state and stopping after the current iteration")
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


def lr_factor(it, total, base_lr, min_lr, warm_fraction):
    warm = max(1, int(total * warm_fraction))
    if it < warm:
        return (it + 1) / warm
    prog = min(1.0, (it - warm) / max(1, total - warm))
    floor = min_lr / base_lr
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))


def rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def set_rng_state(s, device):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if device.type == "cuda" and "cuda" in s:
        torch.cuda.set_rng_state(s["cuda"].cpu(), device)


def env_info(device):
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "host": socket.gethostname(),
            "torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"cuda": torch.version.cuda, "gpu": p.name, "gpu_memory_gb": round(p.total_memory / 2 ** 30, 1)})
    return info


@torch.no_grad()
def evaluate(model, data, mean, std, micro, pool, nsd_tol, precision):
    model.eval()
    loss_sum, rows = 0.0, []
    for i in range(0, data.n, micro):
        idx = torch.arange(i, min(i + micro, data.n))
        x, y = get_batch(data, idx, mean, std)
        with autocast_context(precision, x.device):
            logits = model(prep_input(x, model))
        loss_sum += seg_loss(logits, y)[0].item() * len(idx)
        rows += pool.map(logits.argmax(1).to(torch.uint8).cpu().numpy(), y.to(torch.uint8).cpu().numpy(), nsd_tol)
    model.train()
    a = np.array(rows, dtype=np.float64)
    out = {"loss": loss_sum / data.n}
    for k in EVAL_KEYS:
        out[k] = float(np.nanmean(a[:, METRIC_COLUMNS.index(k)]))
    return out


class Trainer:
    def __init__(self, cfg, layout, model_name, dataset, seed, device, pool, stop, registry=None, worker="main",
                 deadline=None, position=0):
        self.cfg, self.layout = cfg, layout
        self.model_name, self.dataset, self.seed = model_name, dataset, seed
        self.device, self.pool, self.stop = device, pool, stop
        self.reg, self.worker, self.deadline, self.position = registry, worker, deadline, position
        self.key = f"train/{model_name}/{dataset}/seed{seed}"
        self.run_dir = layout.run_dir(model_name, dataset, seed)
        self.log = get_logger()

    def _registry(self, **fields):
        if self.reg is not None:
            self.reg.update(self.key, **fields)

    def _event(self, event, detail=""):
        if self.reg is not None:
            self.reg.event(self.worker, self.key, event, detail)

    def run(self):
        try:
            return self._run()
        finally:
            free_memory(self.device)

    def _run(self):
        cfg, tc = self.cfg, self.cfg.train
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / "DONE").exists():
            self._registry(status="done")
            return "done"
        device = self.device
        tr = load_training_split(self.layout, cfg, self.dataset, "train")
        va = load_training_split(self.layout, cfg, self.dataset, "val")
        norm, mean, std = norm_tensors(self.layout, cfg, self.dataset, device)
        train_data, val_data = GPUData(tr["images"], tr["masks"], device), GPUData(va["images"], va["masks"], device)
        sub_data = train_data.subset(trainsub_indices(self.dataset, train_data.n, tc.trainsub_size))
        del tr, va

        seed_everything(self.seed)
        model = prepare_model(build_model(self.model_name, cfg.data.num_classes, cfg.data.img_size).to(device),
                              tc.channels_last, device)
        n_params = sum(p.numel() for p in model.parameters())
        total = tc.total_iters
        precision = resolve_precision(tc.precision, device)
        scaler = make_scaler(precision, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=tuple(tc.betas))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: lr_factor(it, total, tc.lr, tc.min_lr, tc.warmup_fraction))
        sampler = Sampler(train_data.n, self.seed)
        aug_gen = torch.Generator(device=device)
        aug_gen.manual_seed(self.seed + 100003)
        flags = tc.augmentation.per_dataset[self.dataset]
        micro = tc.micro_batch.get(self.model_name, tc.batch_size)

        start_it, train_time, skipped = 0, 0.0, 0
        best = {"val_dice": -1.0, "iteration": -1, "metrics": None}
        last_metrics, resumed = None, False
        last_p = run_dir / "last.pt"
        if last_p.exists():
            s = torch.load(last_p, map_location="cpu", weights_only=False)
            if not s.get("complete", False) and "optimizer" in s:
                model.load_state_dict(s["model"])
                optimizer.load_state_dict(s["optimizer"])
                scheduler.load_state_dict(s["scheduler"])
                scaler.load_state_dict(s["scaler"])
                sampler.load(s["sampler"])
                aug_gen.set_state(s["aug_gen"].cpu())
                set_rng_state(s["rng"], device)
                start_it, train_time, skipped = s["iteration"], s["train_time_s"], s["skipped_steps"]
                best, last_metrics, resumed = s["best"], s["last_metrics"], True
            del s
        if not resumed:
            for name in ("train_log.csv", "val_log.csv", "trainsub_log.csv", "best.pt", "last.pt", "summary.json"):
                (run_dir / name).unlink(missing_ok=True)
        for name in ("train_log.csv", "val_log.csv", "trainsub_log.csv"):
            truncate_csv(run_dir / name, start_it)

        config = {"model": self.model_name, **model_info(self.model_name), "dataset": self.dataset, "seed": self.seed,
                  "params": n_params, "n_train": train_data.n, "n_val": val_data.n, "micro_batch": micro,
                  "precision": precision, "norm": norm, "train": tc.to_dict(), "img_size": cfg.data.img_size,
                  "split_manifest_sha256": sha256_file(self.layout.split_manifest(self.dataset)),
                  "test_split_used": False}
        save_json({"config": config, "env": env_info(device)}, run_dir / "config.json")
        self._registry(status="running", worker=self.worker, iteration=start_it, started=now())
        self._event("resumed" if resumed else "started", str(start_it))
        self.log.info(f"{'resume' if resumed else 'start'} {self.key} at {start_it}/{total} | {n_params / 1e6:.1f}M params | "
                      f"train {train_data.n} val {val_data.n} | batch {tc.batch_size} (micro {micro}) | {precision}")

        def save_resume(it):
            save_torch({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state(),
                        "aug_gen": aug_gen.get_state(), "rng": rng_state(device), "iteration": it,
                        "train_time_s": train_time, "skipped_steps": skipped, "best": best,
                        "last_metrics": last_metrics, "complete": False, "config": config}, last_p)

        model.train()
        acc, acc_n, it = np.zeros(4), 0, start_it
        seg_start = time.time()
        bar = progress(total=total, initial=start_it, desc=f"{self.model_name}/{self.dataset}/s{self.seed}",
                       position=self.position)
        while it < total:
            if self.stop.requested or (self.deadline is not None and time.time() > self.deadline):
                train_time += time.time() - seg_start
                save_resume(it)
                self._registry(status="paused", iteration=it)
                self._event("paused", str(it))
                self.log.info(f"paused {self.key} at iteration {it}; run the same command to resume")
                bar.close()
                return "paused"
            idx = sampler.next(tc.batch_size)
            x, y = get_batch(train_data, idx, mean, std, tc.augmentation, flags, aug_gen)
            stats = self._step(model, x, y, optimizer, scaler, precision, micro, tc.grad_clip)
            scheduler.step()
            it += 1
            bar.update(1)
            if stats is None:
                skipped += 1
                self.log.info(f"{self.key}: non-finite loss at iteration {it}, step skipped")
            else:
                acc += stats
                acc_n += 1
            if it % tc.log_every == 0:
                n = max(acc_n, 1)
                append_csv(run_dir / "train_log.csv", {"iteration": it, "loss_total": acc[0] / n, "loss_dice": acc[1] / n,
                                                       "loss_ce": acc[2] / n, "train_dice": acc[3] / n})
                bar.set_postfix(loss=f"{acc[0] / n:.3f}", tdice=f"{acc[3] / n:.3f}", best=f"{best['val_dice']:.3f}")
                acc, acc_n = np.zeros(4), 0
            if it % tc.val_every == 0 or it == total:
                v = evaluate(model, val_data, mean, std, micro, self.pool, cfg.metrics.nsd_tolerance_px, precision)
                s = evaluate(model, sub_data, mean, std, micro, self.pool, cfg.metrics.nsd_tolerance_px, precision)
                is_best = v["dice"] > best["val_dice"]
                append_csv(run_dir / "val_log.csv", {"iteration": it, **{f"val_{k}": x_ for k, x_ in v.items()},
                                                     "is_best": int(is_best)})
                append_csv(run_dir / "trainsub_log.csv", {"iteration": it, **{f"trainsub_{k}": x_ for k, x_ in s.items()}})
                last_metrics = {"iteration": it, **v}
                if is_best:
                    best = {"val_dice": v["dice"], "iteration": it, "metrics": last_metrics}
                    save_torch({"model": model.state_dict(), "model_name": self.model_name, "dataset": self.dataset,
                                "seed": self.seed, "iteration": it, "val_metrics": last_metrics, "norm": norm,
                                "img_size": cfg.data.img_size, "precision": precision}, run_dir / "best.pt")
                bar.set_postfix(val=f"{v['dice']:.3f}", best=f"{best['val_dice']:.3f}@{best['iteration']}")
                self._registry(iteration=it, best_iteration=best["iteration"], best_val_dice=best["val_dice"])
                self.log.info(f"{self.key} it {it}/{total} | val dice {v['dice']:.4f} iou {v['iou']:.4f} hd95 {v['hd95']:.2f} "
                              f"| trainsub dice {s['dice']:.4f} | best {best['val_dice']:.4f}@{best['iteration']}"
                              f"{' *' if is_best else ''}")
            if it % cfg.runtime.cleanup_every == 0:
                free_memory(device)
            if it < total and it % tc.resume_every == 0:
                train_time += time.time() - seg_start
                seg_start = time.time()
                save_resume(it)
        bar.close()
        train_time += time.time() - seg_start
        save_torch({"model": model.state_dict(), "model_name": self.model_name, "dataset": self.dataset, "seed": self.seed,
                    "iteration": it, "val_metrics": last_metrics, "norm": norm, "img_size": cfg.data.img_size,
                    "precision": precision, "complete": True}, last_p)
        save_json({"model": self.model_name, "dataset": self.dataset, "seed": self.seed, "params": n_params,
                   "best_iteration": best["iteration"], "best_val_dice": best["val_dice"],
                   "best_val_metrics": best["metrics"], "final_val_metrics": last_metrics,
                   "train_time_h": train_time / 3600.0, "skipped_steps": skipped, "finished": now()},
                  run_dir / "summary.json")
        (run_dir / "DONE").write_text(now())
        self._registry(status="done", iteration=it, best_val_dice=best["val_dice"], best_iteration=best["iteration"],
                       train_time_h=round(train_time / 3600.0, 3), finished=now())
        self._event("finished", f"best_val_dice={best['val_dice']:.4f}@{best['iteration']}")
        self.log.info(f"done {self.key}: best val dice {best['val_dice']:.4f} at {best['iteration']} ({train_time / 3600:.2f} h)")
        return "done"

    @staticmethod
    def _step(model, x, y, optimizer, scaler, precision, micro, clip):
        optimizer.zero_grad(set_to_none=True)
        b = x.shape[0]
        stats = np.zeros(4)
        for xs, ys in zip(x.split(micro), y.split(micro)):
            with autocast_context(precision, xs.device):
                logits = model(prep_input(xs, model))
            loss, ld, lc = seg_loss(logits, ys)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                return None
            w = xs.shape[0] / b
            scaler.scale(loss * w).backward()
            with torch.no_grad():
                td = hard_dice(logits.argmax(1), ys).sum().item()
            stats += np.array([loss.item() * w, ld.item() * w, lc.item() * w, td / b])
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        return stats
