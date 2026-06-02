# Twister Body Control Benchmark

Benchmark for testing how well an AI can operate a human body in a physics simulator. The agent controls a 3D rigged humanoid via **position-controlled joint angles** (Option B) in MuJoCo and plays **Twister**.

Built on [Mesocosm / BenchAnything](https://github.com/swecc-io/mesocosm) — runs as an HTTP env that LLM agents call over `reset` / `step`.

CLICK HERE FOR DEMO: https://shivani-kulandaivel.github.io/twisterBenchmark/

## What it tests

- **Spatial reasoning** — mapping spinner commands to body configurations
- **Embodied control** — choosing joint angles to reach colored circles on a 4×6 mat
- **Balance under constraint** — staying upright while holding awkward poses (Phase 2)

## Phases

| Phase | Task | Episode ends when |
|-------|------|-------------------|
| **1** | Single limb placement (`"Place left hand on green"`) | Success, fall, or 50 steps |
| **2** | Multi-turn survival — locked limbs must stay put | Fall, slipped limb, 10 turns, or 200 steps |

Select phase via `reset(params={"phase": 1})` or `{"phase": 2}`.

## Control model

The agent sends **target joint angles in degrees**. MuJoCo position actuators (PD controllers) track them. No torque control in v1.

**Note:** v1 uses a fixed pelvis (no floating base) so the figure stays upright during reset. Falls are detected via excessive torso tilt from bad poses. A floating-base balance mode may be added later as a harder tier.

### Action

```json
{
  "joint_targets": {
    "left_shoulder_pitch": 45.0,
    "left_elbow": 90.0
  },
  "delta": false
}
```

- Omitted joints hold their previous target.
- Set `"delta": true` to send per-step angle changes (capped at ±15°).

### Observation (abbreviated)

```json
{
  "phase": 1,
  "command": { "limb": "left_hand", "color": "green", "circle": [2, 3], "instruction": "..." },
  "joints": { "left_elbow": 10.0 },
  "end_effectors": { "left_hand": { "x": 0.3, "y": 0.0, "z": 1.1, "on_mat": false } },
  "mat": { "rows": 4, "cols": 6, "circles": [] },
  "placement_error": 0.42,
  "placement_radius": 0.08,
  "upright": true
}
```

### Controllable joints (15 DoF)

`abdomen_pitch`, `left/right_shoulder_pitch`, `left/right_shoulder_roll`, `left/right_elbow`, `left/right_hip_pitch`, `left/right_hip_roll`, `left/right_knee`, `left/right_ankle`

## Setup

```bash
pip install swecc-mesocosm
pip install -r requirements.txt
```

## Local dev (Ollama)

See **[`LOCAL_DEV.md`](LOCAL_DEV.md)** and the [Mesocosm local development wiki](https://wiki.swecc.org/Sweccathon/mesocosm/local-development).

**One-time:** install [Ollama](https://ollama.com), then `ollama pull llama3.2`.

**Terminal 1 — env server** (run from the repo root)

```powershell
pip install -r requirements.txt
python adapter.py
```

**Terminal 2 — run episodes** (run from the repo root)

```powershell
$env:PYTHONUTF8 = "1"
mesocosm run local --model ollama/llama3.2 --episodes 3 --max-tokens 1024 --system-prompt (Get-Content system_prompt.txt -Raw)
```

## Project layout

The env lives at the repo root so the platform finds `benchanything.json` there.

```
benchanything.json      # Manifest + scoring (at repo root)
env.py                  # TwisterEnv (reset/step)
adapter.py              # HTTP adapter
sim/
├── humanoid.py         # MuJoCo wrapper
├── constants.py        # Joint schema
└── mjcf/humanoid_twister.xml
game/
├── twister.py          # Mat, spinner, validation
└── tasks.py            # Phase 1 & 2 logic
showcase/               # Replay viewer (deployed to GitHub Pages)
```

## Scoring

- **Phase 1:** pass rate (reward ≥ 1.0 = correct placement while upright)
- **Phase 2:** mean `turns_survived` in episode info
- **Both:** mean `avg_placement_error` (meters to target circle)

## Ship to Mesocosm

```bash
mesocosm auth login
mesocosm env submit --name "Twister Body" --github-url https://github.com/you/twisterBenchmark
```
