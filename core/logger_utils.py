"""
core/logger_utils.py

Run logging helpers for configs, stage events, and JSON manifests.
"""

import json
import os
import platform
import sys
import time

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

#Logger utils:

def _utc_now():
    """Internal helper for utc now."""
    return datetime.now(timezone.utc).isoformat()

def _to_jsonable(x):
    """Recursively convert common non-JSON types (numpy/torch/etc.) to JSON-safe."""
    # basic
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # dict-like
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # numpy scalars/arrays
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
    except Exception:
        pass

    # torch tensors/devices
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
            }
        if isinstance(x, torch.device):
            return str(x)
    except Exception:
        pass

    # pathlib paths
    try:
        from pathlib import Path
        if isinstance(x, Path):
            return str(x)
    except Exception:
        pass

    # fallback: stringify (handles callables, classes, etc.)
    return {"__type__": type(x).__name__, "__repr__": repr(x)}

class RunLogger:
    """JSON-backed logger for run configs and high-level stage events."""
    def __init__(self, out_dir: str, filename: str = "run_manifest.json"):
        """Store configuration and initialize the runtime state for this object."""
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, filename)

        self.data = {
            "created_utc": _utc_now(),
            "out_dir": out_dir,
            "env": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "configs": {},
            "stages": [],   # chronological list of stage records
        }
        self.flush()  # create file immediately

    def set_config(self, key: str, cfg_obj):
        """Attach the final resolved configuration to the run log."""
        self.data["configs"][key] = _to_jsonable(cfg_obj)
        self.flush()

    def flush(self):
        """Write the current run log state to disk."""
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=False)
        os.replace(tmp, self.path)  # atomic on POSIX/most OSes

    @contextmanager
    def stage(self, name: str, **meta):
        """Record a named stage transition in the run log."""
        rec = {
            "name": name,
            "start_utc": _utc_now(),
            "meta": _to_jsonable(meta),
        }
        t0 = time.perf_counter()
        try:
            yield rec  # you can mutate rec inside the with-block to add outputs
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = {"type": type(e).__name__, "repr": repr(e)}
            raise
        finally:
            rec["end_utc"] = _utc_now()
            rec["duration_s"] = float(time.perf_counter() - t0)
            self.data["stages"].append(rec)
            self.flush()
