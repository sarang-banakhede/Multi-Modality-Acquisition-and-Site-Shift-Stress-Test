import argparse

from .config import load_config
from .pipeline import STAGES, run_pipeline
from .status import print_status


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="shift-stress-test", description="Robustness of medical image segmentation under calibrated degradation and site shift")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--stages", nargs="+", choices=STAGES)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE")
    ap.add_argument("--status", action="store_true", help="print progress of every job and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    cfg = load_config(args.config, args.overrides)
    if args.status:
        print_status(cfg, args.models, args.datasets, args.seeds)
        return
    run_pipeline(cfg, args.stages, args.models, args.datasets, args.seeds)


if __name__ == "__main__":
    main()
