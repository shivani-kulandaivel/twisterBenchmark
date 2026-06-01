"""Rendering benchmark measurements."""

from __future__ import annotations

import time

import mujoco

from sim.humanoid import HumanoidSim


def run_render_bench(frames: int = 240) -> dict[str, float]:
    sim = HumanoidSim()
    sim.reset(seed=0)
    renderer = mujoco.Renderer(sim.model, height=720, width=1280)
    frame_times: list[float] = []
    for _ in range(frames):
        sim.step_physics(substeps=1)
        t0 = time.perf_counter()
        renderer.update_scene(sim.data)
        renderer.render()
        frame_times.append((time.perf_counter() - t0) * 1000.0)
    frame_times_sorted = sorted(frame_times)
    p99 = frame_times_sorted[int(0.99 * (len(frame_times_sorted) - 1))]
    avg = sum(frame_times) / len(frame_times)
    return {
        "avg_fps": 1000.0 / max(1e-6, avg),
        "p99_frame_ms": p99,
        "render_time_per_frame_ms": avg,
        "draw_calls": float(sim.model.ngeom),
        "triangle_count": float(sim.model.nmeshface),
        "dropped_frames": float(sum(1 for x in frame_times if x > 16.67)),
    }
