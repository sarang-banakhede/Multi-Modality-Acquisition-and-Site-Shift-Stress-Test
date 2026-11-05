import logging
import sys
from pathlib import Path

from tqdm import tqdm

NAME = "shift_stress_test"


class TqdmHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def setup_logging(log_file=None, tag="main"):
    logger = logging.getLogger(NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(f"[%(asctime)s] [{tag}] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = TqdmHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger():
    return logging.getLogger(NAME)


def progress(iterable=None, **kw):
    kw.setdefault("dynamic_ncols", True)
    kw.setdefault("leave", False)
    return tqdm(iterable, **kw)
