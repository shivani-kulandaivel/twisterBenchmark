"""CLI for benchmark suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.agent_bench import run_agent_bench
from benchmark.ai_bench import run_ai_bench
from benchmark.memory_bench import run_memory_bench
from benchmark.physics_bench import run_physics_bench
from benchmark.render_bench import run_render_bench


def run_suite(name: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if name in {"all", "physics"}:
        out["physics"] = run_physics_bench()
    if name in {"all", "render"}:
        out["render"] = run_render_bench()
    if name in {"all", "ai"}:
        out["ai"] = run_ai_bench()
    if name in {"all", "memory"}:
        out["memory"] = run_memory_bench()
    if name in {"all", "agent"}:
        out["agent"] = run_agent_bench()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="all", choices=["all", "physics", "render", "ai", "memory", "agent"])
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    report = run_suite(args.suite)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.suite}_bench.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
