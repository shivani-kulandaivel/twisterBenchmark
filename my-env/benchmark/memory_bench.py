"""Memory usage benchmark."""

from __future__ import annotations

import gc
import tracemalloc

from env import TwisterEnv


def run_memory_bench(episodes: int = 20) -> dict[str, float]:
    gc.collect()
    tracemalloc.start()
    before = len(gc.get_objects())
    for ep in range(episodes):
        env = TwisterEnv()
        env.reset(seed=ep, phase=1)
        env.step({"use_ik": True})
    current, peak = tracemalloc.get_traced_memory()
    after = len(gc.get_objects())
    gc_stats = gc.get_stats()
    tracemalloc.stop()
    collections = float(sum(gen["collections"] for gen in gc_stats))
    return {
        "ram_usage_bytes": float(current),
        "peak_rss_mb": peak / (1024 * 1024),
        "object_count": float(after - before),
        "gc_collections": collections,
        "vram_usage_mb": 0.0,
    }
