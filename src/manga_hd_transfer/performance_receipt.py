"""Cheap host performance counters shared by CLI and GUI page runs."""
from __future__ import annotations

import platform
import threading
import time

import cv2


def sample_process() -> dict:
    sample = {"wall": time.perf_counter(), "cpu": time.process_time(),
              "thread": threading.current_thread().name,
              "opencv_threads": int(cv2.getNumThreads())}
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        sample["peak_rss_bytes"] = int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
        sample["major_faults"] = int(usage.ru_majflt)
    except (ImportError, OSError, ValueError):
        pass
    try:
        import psutil
        sample["rss_bytes"] = int(psutil.Process().memory_info().rss)
        sample["available_memory_bytes"] = int(psutil.virtual_memory().available)
    except ImportError:
        pass
    except (OSError, psutil.Error):
        pass
    return sample


def performance_delta(start: dict, end: dict) -> dict:
    return {"schema": "folirina.host_performance.v1", "thread": end["thread"],
            "wall_seconds": round(max(0.0, end["wall"] - start["wall"]), 6),
            # Main process CPU time excludes isolated model workers. Do not
            # interpret CPU/wall as GPU utilization or a fallback diagnosis.
            "main_process_cpu_seconds": round(max(0.0, end["cpu"] - start["cpu"]), 6),
            "opencv_effective_threads": end["opencv_threads"],
            "rss_start_bytes": start.get("rss_bytes"), "rss_end_bytes": end.get("rss_bytes"),
            "peak_process_rss_bytes": end.get("peak_rss_bytes"),
            "available_memory_end_bytes": end.get("available_memory_bytes"),
            "major_faults_delta": max(0, end.get("major_faults", 0) - start.get("major_faults", 0))}
