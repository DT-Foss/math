from __future__ import annotations
import json, time
import numpy as np


def median_ms(fn, repeats=5):
    times=[]
    for _ in range(repeats):
        t=time.perf_counter(); fn(); times.append((time.perf_counter()-t)*1000)
    return float(np.median(times))


def emit(obj):
    print(json.dumps(obj,sort_keys=True))
