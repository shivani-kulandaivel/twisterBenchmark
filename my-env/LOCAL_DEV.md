# Local development (Ollama)

Iterate on `env.py` and `benchanything.json` on your machine before `mesocosm env submit`. Local runs use **Ollama** — no cloud API keys required.

Official docs: [Mesocosm local development](https://wiki.swecc.org/Sweccathon/mesocosm/local-development)

## Prerequisites

### 1. Mesocosm CLI

```powershell
pip install swecc-mesocosm
```

This includes `mesocosm run local`, `bench_common` (for `adapter.py`), and the HTTP stack (`fastapi`, `uvicorn`).

### 2. Environment dependencies (Twister)

This benchmark imports MuJoCo, so install extra deps locally:

```powershell
cd my-env
pip install -r requirements.txt
```

(`mujoco`, `numpy` — the platform installs these on `env submit` too.)

### 3. Ollama

1. Install from [ollama.com](https://ollama.com) (Windows/macOS/Linux desktop app).
2. **Close and reopen PowerShell** after install so PATH updates.
3. If `ollama` is still not recognized, it is usually already installed here:

   ```powershell
   & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
   ```

   Add to your user PATH permanently:

   ```powershell
   [Environment]::SetEnvironmentVariable(
     "PATH",
     [Environment]::GetEnvironmentVariable("PATH", "User") + ";$env:LOCALAPPDATA\Programs\Ollama",
     "User"
   )
   ```

   Then open a **new** PowerShell window.

4. Pull a model:

   ```powershell
   ollama pull llama3.2
   ```

5. Ensure Ollama is running — the desktop app usually starts the server. Otherwise:

   ```powershell
   ollama serve
   ```

6. Verify:

   ```powershell
   ollama list
   ```

## Dev loop

Optional but recommended:

```powershell
$env:MESOCOSM_LOCAL = "1"
# Windows: avoid Unicode errors in mesocosm CLI output
$env:PYTHONUTF8 = "1"
cd my-env
mesocosm doctor --local
```

`doctor --local` checks that your adapter responds on port **8765**.

### Terminal 1 — env adapter

```powershell
cd my-env
python adapter.py
# Health check: http://localhost:8765/health
```

### Terminal 2 — benchmark episodes

```powershell
cd my-env
mesocosm run local
```

Equivalent to:

```powershell
mesocosm run local --model ollama/llama3.2
```

`run local`:

- Reads `benchanything.json` for the binding vow and scoring rules
- Calls your adapter at `http://localhost:8765` by default
- Uses an Ollama model (`ollama/…` prefix **required**)
- Does **not** register the domain or create platform runs

### Twister-specific run (recommended)

Use the bundled system prompt so the model outputs valid joint JSON:

```powershell
cd my-env
$env:PYTHONUTF8 = "1"
mesocosm run local `
  --model ollama/llama3.2 `
  --episodes 3 `
  --max-tokens 1024 `
  --system-prompt (Get-Content system_prompt.txt -Raw)
```

Or paste a shorter prompt inline:

```powershell
mesocosm run local --episodes 3 --max-tokens 1024 --system-prompt "Respond with JSON only: {\"joint_targets\": {\"left_elbow\": 90}}. Angles are degrees. Follow command.instruction in the observation to move the named limb toward target_circle."
```

**Phase 2** (multi-turn survival) — pass seed/params via env reset is automatic; to test Phase 2 you'd extend the env or use params in a custom runner. Phase 1 is the default (`reset` uses `phase=1`).

## Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `ollama/llama3.2` | LiteLLM model id; must start with `ollama/` and match a pulled model |
| `--episodes` | `5` | Number of episodes |
| `--env-url` | `http://localhost:8765` | Adapter base URL if you changed the port |
| `--manifest` | `benchanything.json` | Alternate manifest path |
| `--domain-id` | from manifest or folder name | Override domain id for local runs |
| `--system-prompt` | — | Extra instruction for the agent (strongly recommended for Twister) |
| `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `512` | Max tokens per step (use **1024** for joint JSON) |
| `--parallel` | `1` | Max parallel episodes |
| `--seeds` | — | Space-separated integer seeds, e.g. `--seeds 1 2 3` |
| `--quiet` | — | Less progress output |

## Quick sanity check (no Ollama)

Verify the sim loads before running the full agent loop:

```powershell
cd my-env
python -c "from env import TwisterEnv; e=TwisterEnv(); o=e.reset(seed=42); print(o['command']['instruction']); r=e.step({'joint_targets':{'left_shoulder_pitch':45}}); print('reward', r.reward, 'upright', r.observation['upright'])"
```

## Ship to Mesocosm

When local runs look good:

```powershell
mesocosm auth login
mesocosm env submit --name "Twister Body" --github-url https://github.com/you/twisterBenchmark
mesocosm env list
mesocosm run create --domain DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite --episodes 5
```

Platform runs use cloud models on SWECC infrastructure. **Ollama is only for your machine.**

**Non-interactive auth:** set `SWECC_BENCH_TOKEN` or use `mesocosm auth guest`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ollama` not recognized | Ollama is often installed but PATH not refreshed. Run `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version`. Add `%LOCALAPPDATA%\Programs\Ollama` to user PATH, then **open a new terminal** |
| `Cannot connect to host localhost:11434` | Ollama server not running — open the Ollama app or run `ollama serve` |
| `model requires more system memory` | `llama3.2` needs ~2.3 GiB free RAM. Use a smaller model: `ollama pull llama3.2:1b` then `mesocosm run local --model ollama/llama3.2:1b`. Close other apps; MuJoCo also uses RAM while the adapter runs |
| `UnicodeEncodeError` in mesocosm | Run `$env:PYTHONUTF8 = "1"` before `mesocosm run local` (Windows) |
| Connection refused on 8765 | Start `python adapter.py` in Terminal 1 first |
| `ModuleNotFoundError: mujoco` | Run `pip install -r requirements.txt` in `my-env` |
| Model returns prose instead of JSON | Use `--system-prompt` from `system_prompt.txt`; try `--temperature 0.0` |
| `ollama/llama3.2` not found | Run `ollama pull llama3.2` or pass `--model ollama/<your-model>` |

## Related

- [Mesocosm getting started](https://wiki.swecc.org/Sweccathon/mesocosm/)
- [Command reference — run local](https://wiki.swecc.org/Sweccathon/mesocosm/)
- Root [`README.md`](../README.md) — observation/action schemas
