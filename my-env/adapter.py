"""
HTTP adapter — exposes TwisterEnv via the BenchAnything four-endpoint protocol.

Local dev:
    python adapter.py
    python adapter.py --port 9000

The adapter injects system_prompt on reset and exposes GET /schema for agents.
"""

from __future__ import annotations

import argparse
from typing import Any

from bench_common.env_sdk.base import BaseEnv
from env import TwisterEnv, _load_system_prompt, action_schema_for_prompt, joint_schema_for_prompt


class TwisterAdapterEnv(TwisterEnv):
    """Wraps reset observations with system_prompt for Mesocosm HTTP clients."""

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        data = super().reset(seed=seed, **params)
        return {
            "data": data,
            "content_type": "application/json",
            "system_prompt": _load_system_prompt(),
        }

    def step(self, action: Any) -> StepResult:
        result = super().step(action)
        if isinstance(result.observation, dict):
            return result
        return result


def _install_schema_route(app: Any) -> None:
    from fastapi import FastAPI

    if not isinstance(app, FastAPI):
        return

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        return {
            "joints": joint_schema_for_prompt(),
            "actions": action_schema_for_prompt(),
            "observation_notes": {
                "active_limb_goal": (
                    "Exact placement objective for this step: control point name, "
                    "target_xyz, current_xyz, and error_xyz."
                ),
                "placement_contract": (
                    "Authoritative success metric. Success when horizontal distance "
                    "between control_point.xy and target_center.xy is <= placement_radius."
                ),
            },
            "control_notes": (
                "Whole-body reach uses operational-space IK (Khatib/Sentis-style hierarchy): "
                "end-effector tracking, null-space balance, posture regularization. "
                "Use observation.ik_suggestion as a physics-aware reference pose, "
                "or set action.use_ik=true to delegate one step to the built-in controller. "
                "Advanced: action.controller writes your JS/Python script to "
                "generated_controllers/ and executes it each prompt."
            ),
        }


def serve_twister(**kwargs: Any) -> None:
    """Start adapter with Twister-specific schema endpoint."""
    from bench_common.env_sdk import server as sdk_server

    env_class = TwisterAdapterEnv
    host = kwargs.get("host", "0.0.0.0")
    port = kwargs.get("port", 8765)
    log_level = kwargs.get("log_level", "info")

    _episodes: dict[str, BaseEnv] = {}

    from bench_common.env_sdk.server import (
        _CloseRequest,
        _RenderRequest,
        _ResetRequest,
        _StepRequest,
    )
    from fastapi import FastAPI, HTTPException
    import logging
    import uvicorn

    log = logging.getLogger(__name__)
    app = FastAPI(title="Twister Body Control Adapter", version="1.1.0")
    _install_schema_route(app)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "env": env_class.__name__, "episodes": len(_episodes)}

    @app.post("/reset")
    def reset(req: _ResetRequest) -> dict:
        if req.episode_id in _episodes:
            try:
                _episodes[req.episode_id].close()
            except Exception:
                pass
        env = env_class()
        _episodes[req.episode_id] = env
        try:
            obs = env.reset(seed=req.seed, **req.scenario_params)
        except Exception as exc:
            _episodes.pop(req.episode_id, None)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if isinstance(obs, dict) and "data" in obs:
            return obs
        return {"data": obs, "content_type": "application/json", "system_prompt": _load_system_prompt()}

    @app.post("/step")
    def step(req: _StepRequest) -> dict:
        env = _episodes.get(req.episode_id)
        if env is None:
            raise HTTPException(status_code=404, detail="Call /reset first.")
        try:
            result = env.step(req.action)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "observation": {"data": result.observation, "content_type": "application/json"},
            "reward": float(result.reward),
            "terminated": bool(result.terminated),
            "truncated": bool(result.truncated),
            "info": {str(k): str(v) for k, v in result.info.items()},
            "system_prompt": result.system_prompt,
        }

    @app.post("/close")
    def close(req: _CloseRequest) -> dict:
        env = _episodes.pop(req.episode_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return {}

    @app.post("/render")
    def render(req: _RenderRequest) -> dict:
        env = _episodes.get(req.episode_id)
        if env is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        data = env.render(mode=req.mode)
        content_type = "text/plain" if isinstance(data, str) else "application/json"
        return {"data": data, "content_type": content_type}

    log.info("Starting TwisterAdapterEnv on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"TwisterEnv adapter -> http://{args.host}:{args.port}")
    print(f"  GET /schema  — joint limits, action format, control notes")
    serve_twister(host=args.host, port=args.port)
