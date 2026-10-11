# Architecture — PotatoServe

## 1. Overview
PotatoServe is a thin orchestration and measurement layer around llama.cpp. It does not reimplement inference: it **controls** it, **measures** it, and — in the `lab/` track — **improves** it upstream.

## 2. System diagram
```
                 ┌──────────────────────────┐
                 │      potato CLI          │
                 └────────────┬─────────────┘
    ┌──────────┬──────────┬───┴─────┬───────────┬──────────┐
  probe      quant      bench     tune/serve   report     lab
 (HW info)  (GGUF)   (speed+qual)  (search)   (md/json)  (upstream)
    │          │          │           │           │          │
    └──────────┴────┬─────┴───────────┘           │          │
                    ▼                             ▼          ▼
         ┌─────────────────────┐          results.sqlite   fork of
         │ runner (subprocess) │◄──────── + JSON records   llama.cpp
         └──────────┬──────────┘                           (patches)
                    ▼
   llama-bench │ llama-perplexity │ llama-server ◄── agent-eval client
        (pinned llama.cpp build: CUDA / Vulkan / CPU)
```

## 3. Components
| Component | Responsibility |
|-----------|----------------|
| probe | Detect GPU/VRAM/driver/backends/CPU flags (NVML, `vulkaninfo`, psutil) |
| quant | HF → GGUF conversion, imatrix generation, quantize variants |
| runner | Start/stop llama.cpp processes, build arg lists from config, capture output, enforce timeouts, sample VRAM via NVML |
| bench | Speed, memory, quality (KL / perplexity) with repeats |
| agent-eval | Task harness: sandbox dir, OpenAI-compatible client, checkers |
| tune | Memory-model pruning + staged search → Pareto set |
| report | Tables, charts, environment fingerprint |
| store | SQLite + JSON; append-only |
| lab | Fork + experiments for upstream contributions |

## 4. Tuner strategy
1. **Prune** configs that cannot fit using the memory model.
2. **Stage A (coarse):** speed at short context for all survivors.
3. **Stage B:** top-N from A → add context depth sweep + KL gate (reject above threshold).
4. **Stage C:** finalists → agent-eval.
5. Emit a **Pareto set** over (decode tok/s, quality, max context) and a recommended config per objective.

## 5. Key decisions (ADR-lite)
- **ADR-1 Wrap, don't fork.** Fork only inside `lab/` for upstream patches.
- **ADR-2 Pin everything.** Every result stores llama.cpp commit + build flags; cross-build comparisons are flagged.
- **ADR-3 KL divergence over perplexity alone.** Perplexity can hide behavior drift that agents are sensitive to.
- **ADR-4 Python for orchestration, C++/CUDA only for upstream work.**
- **ADR-5 Append-only results.** Never overwrite a run; supersede with a new one.

## 6. Failure modes & handling
| Failure | Handling |
|---------|----------|
| OOM / driver reset mid-run | Runner catches exit code + log signature; marks config `infeasible`, continues |
| Server hang (known class of long-context hangs) | Watchdog timeout; kill; record `hang` with logs |
| Thermal throttling skews numbers | Sample clocks/temps; discard or flag runs outside tolerance |
| Non-determinism (e.g. spec decoding) | Repeat runs; report spread; never claim differences inside noise |
| Upstream change breaks harness | CI job against pinned commit + weekly canary against master |

## 7. Extension points
New backends (Metal, SYCL, Hexagon) = new probe + runner profile; new metrics = plugin functions returning dict; new tasks = YAML + checker.
