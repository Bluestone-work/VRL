"""Observe gradient health without changing the baseline training update."""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import time

import torch


def main():
    path = Path(os.environ["RESEARCH_HEALTH_PATH"])
    original = torch.nn.utils.clip_grad_norm_
    calls = 0
    started = time.monotonic()
    with path.open("x") as log:
        def observed_clip(*args, **kwargs):
            nonlocal calls
            norm = original(*args, **kwargs)
            value = float(norm.detach().cpu())
            calls += 1
            finite = bool(torch.isfinite(norm).item())
            log.write(json.dumps({"clip_call": calls, "pre_clip_norm": value,
                                  "finite": finite, "seconds": time.monotonic() - started},
                                 allow_nan=False) + "\n")
            if calls % 100 == 0:
                log.flush()
            if not finite:
                raise FloatingPointError("Non-finite gradient norm; experiment stopped")
            return norm
        torch.nn.utils.clip_grad_norm_ = observed_clip
        try:
            runpy.run_path(str(Path(__file__).with_name("train_vector_mappo.py")), run_name="__main__")
        finally:
            torch.nn.utils.clip_grad_norm_ = original
            log.flush()
            memory = {}
            if torch.cuda.is_initialized():
                for device in range(torch.cuda.device_count()):
                    memory[str(device)] = {"allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                                           "reserved_peak_bytes": torch.cuda.max_memory_reserved(device)}
            path.with_suffix(".summary.json").write_text(json.dumps(
                {"gradient_calls": calls, "seconds": time.monotonic() - started,
                 "gpu_memory": memory}, indent=2))


if __name__ == "__main__":
    main()
