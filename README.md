# PotatoServe

Auto-tune [llama.cpp](https://github.com/ggml-org/llama.cpp) for low-VRAM GPUs, with
honest speed / memory / quality measurements and a reproducible results store.

> **Status: Phase 0 (foundations) — not yet usable.** See `docs/phases.md`.

## What it is
A thin orchestration and measurement layer around llama.cpp. PotatoServe does not
reimplement inference; it **controls** it, **measures** it, and (in the `lab/` track)
**improves** it upstream. See `docs/architecture.md`.

## Docs
- `docs/prd.md` — why this exists, goals, success metrics
- `docs/project-spec.md` — targets, stack, config space, repo layout (section 9)
- `docs/architecture.md` — components and decisions
- `docs/phases.md` — phase plan and exit criteria
- `docs/memory.md` — living project memory (read at session start, update at end)

## Layout
```
potato/            # python package (CLI + runner + store)
third_party/llama.cpp  # git submodule, pinned commit
lab/               # upstream experiments, patches, perf notes
evals/agent_tasks/ # agent task definitions + checkers
runs/              # gitignored raw run output
results/           # curated/published tables + JSON
docs/ tests/ scripts/
```

## Development
Phase 0 uses only the Python standard library. Dependencies are added one at a
time after explicit approval (project rules). See `docs/memory.md`.

```bash
# run the layout test (no third-party deps required)
python -m unittest discover -s tests -v
```

## License
MIT (matches llama.cpp). See `LICENSE`. Model weights keep their own licenses and are
never redistributed.
