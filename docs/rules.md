# Rules — PotatoServe

> ⚠️ **PENDING CONFIRMATION.** This file was referenced in the session brief
> ("follow rules.md strictly") but was not present on disk. The rules below are
> transcribed from the session brief and the docs — **not invented**. Confirm,
> edit, or replace this file; the layout test expects it to exist.

## Session rules (from the brief)
1. **Don't invent numbers.** The assistant cannot run your hardware, so it gives
   exact commands and asks you to paste back real output.
2. **Ask before adding any dependency.** Each new package is added one at a time,
   only after explicit approval.
3. **Make small commits.**
4. **Verify against the pinned build, not memory** — llama.cpp flag names change
   between commits.
5. **Update `docs/memory.md` at the end of every session** (status, decisions,
   environment fingerprint, next action).

## Project rules (from docs)
- **Wrap, don't fork** llama.cpp (ADR-1). Fork only inside `lab/`.
- **Pin everything** (ADR-2): every result records llama.cpp commit + build flags;
  cross-build comparisons are flagged.
- **Honest measurement** (PRD §7): ≥ 3 runs per point, report median + spread,
  no cherry-picking. Never claim differences inside noise (architecture §6).
- **Append-only results** (ADR-5): never overwrite a run; supersede with a new one.
- **KL divergence as the quality filter**, agent-eval as the final gate (ADR-3).
- **No unexplained numbers** anywhere (phases.md, definition of done).
- **MIT licensing**; never redistribute model weights (project-spec §11).
