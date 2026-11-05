import multiprocessing as mp

import yaml

from .config import Config

from .data.cache import build_cache, ensure_norm
from .data.records import external_ref, internal_ref
from .degrade.build import run_degrade
from .inference import run_inference
from .inference.gates import check_repro_train
from .layout import Layout
from .models import build_model
from .training import run_training
from .utils.log import get_logger, setup_logging

STAGES = ["prepare", "train", "degrade", "infer_val", "freeze", "infer_test", "analyze"]


def stage_prepare(cfg, layout, datasets, models=()):
    for d in datasets:
        for split in ("train", "val", "test"):
            build_cache(layout, cfg, internal_ref(d, split))
        build_cache(layout, cfg, external_ref(cfg, d))
        ensure_norm(layout, cfg, d)
    for m in [m for m in models if m.endswith("_pt")]:
        build_model(m, cfg.data.num_classes, cfg.data.img_size)
        get_logger().info(f"pretrained weights ready: {m}")


def _degrade_worker(cfg_dict, datasets):
    cfg = Config(cfg_dict)
    layout = Layout(cfg)
    setup_logging(layout.log_file, tag="degrade")
    run_degrade(cfg, layout, datasets)


def run_pipeline(cfg, stages=None, models=None, datasets=None, seeds=None):
    layout = Layout(cfg)
    layout.ensure()
    setup_logging(layout.log_file)
    log = get_logger()
    models = models or list(cfg.experiment.models) + list(cfg.experiment.get("pretrained_models", []))
    main = [m for m in models if not m.endswith("_pt")]
    datasets = datasets or list(cfg.experiment.datasets)
    seeds = seeds if seeds is not None else list(cfg.experiment.seeds)
    stages = stages or list(cfg.stages)
    with open(layout.out / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    log.info(f"output {layout.out} | models {len(models)} datasets {len(datasets)} seeds {len(seeds)} | stages {stages}")
    background = None
    if cfg.runtime.overlap_degrade and "train" in stages and "degrade" in stages and stages.index("train") < stages.index("degrade"):
        background = mp.get_context("spawn").Process(target=_degrade_worker, args=(cfg.to_dict(), datasets))
        background.start()
        log.info("degradation runs in the background while the models train")
    for stage in stages:
        log.info(f"=== stage: {stage}")
        if stage == "degrade" and background is not None:
            background.join()
            if background.exitcode != 0:
                raise RuntimeError("background degradation failed, see the log")
            continue
        if stage == "prepare":
            stage_prepare(cfg, layout, datasets, models)
        elif stage == "train":
            run_training(cfg, layout, models, datasets, seeds)
        elif stage == "degrade":
            run_degrade(cfg, layout, datasets)
        elif stage == "infer_val":
            run_inference(cfg, layout, "infer_val", main, datasets, seeds)
            check_repro_train(cfg, layout, main, datasets, seeds)
        elif stage == "freeze":
            from .analysis.freeze import run_freeze
            run_freeze(cfg, layout, main, datasets, seeds)
        elif stage == "infer_test":
            run_inference(cfg, layout, "infer_test", models, datasets, seeds)
        elif stage == "analyze":
            from .analysis.run import run_analysis
            run_analysis(cfg, layout, main, datasets, seeds)
        else:
            raise ValueError(f"unknown stage {stage}")
    log.info("pipeline finished")
