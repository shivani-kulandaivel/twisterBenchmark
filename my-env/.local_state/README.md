# Local state (gitignored)

Ephemeral dev/runtime cache lives here. Nothing in this folder (except this README and
`session.json.example`) is tracked by git.

## What to put here

| File | Purpose |
|------|---------|
| `session.json` | Last seed, run config, and notes from your most recent local session |
| `last_run.json` | Optional snapshot of mesocosm run results (scores, episode count) |
| Any other scratch files | Cached context for agents or local tooling |

## Quick start

Copy the template and edit after each run:

```bash
cp session.json.example session.json
```

Example `session.json` fields:

- `last_seed` — integer seed used in the last `mesocosm run local`
- `last_model` — e.g. `ollama/llama3.2`
- `last_episodes` — episode count
- `notes` — free-form notes (what worked, what to try next)

This avoids re-deriving context across sessions; agents and humans can read
`session.json` to pick up where the last run left off.
