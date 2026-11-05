import fcntl
import json
import socket
from datetime import datetime

from .io import append_csv, save_json


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class JobRegistry:
    def __init__(self, state_file, events_file):
        self.path = state_file
        self.events = events_file
        self.lock = state_file.with_suffix(".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _locked(self, fn):
        with open(self.lock, "a+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() else {"jobs": {}}
                result, changed = fn(data)
                if changed:
                    save_json(data, self.path)
                return result
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def read(self):
        return self._locked(lambda d: (d["jobs"], False))

    def prepare(self, done):
        def fn(d):
            for key, is_done in done.items():
                st = d["jobs"].setdefault(key, {"status": "pending"})
                if is_done:
                    st["status"] = "done"
                elif st["status"] in ("running", "failed"):
                    st["status"] = "paused" if st["status"] == "running" else "pending"
            return None, True
        self._locked(fn)

    def claim(self, keys, worker):
        def fn(d):
            for key in keys:
                st = d["jobs"].setdefault(key, {"status": "pending"})
                if st["status"] in ("done", "running", "failed"):
                    continue
                st.update({"status": "running", "worker": worker, "host": socket.gethostname(), "claimed": now()})
                return key, True
            return None, False
        return self._locked(fn)

    def update(self, key, **fields):
        def fn(d):
            d["jobs"].setdefault(key, {}).update(fields)
            return None, True
        self._locked(fn)

    def event(self, worker, key, event, detail=""):
        append_csv(self.events, {"time": now(), "worker": worker, "job": key, "event": event, "detail": detail})
