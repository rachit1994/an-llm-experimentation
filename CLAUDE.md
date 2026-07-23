# CLAUDE.md

The operating rules for this repo are in **[AGENTS.md](AGENTS.md)** — read it first and follow it
literally. It is the single source of truth for how work is done here; this file only adds a few
Claude-Code-specific notes and imports it.

@AGENTS.md

## Claude-Code-specific notes

- **The one rule that outranks "make it green":** a test is evidence only if it has been shown to fail
  on broken code. Before you report a passing metric/known-answer test, confirm its expected value is
  computed *independently* of the code under test and that it has a paired anti-vacuity canary
  (AGENTS.md rules 1–3). This is where the last serious defect came from; do not repeat it.
- **Provenance, not prose:** run `cargo run -p qilm-train --bin report` to see results. Do not write
  measured numbers into any `.md` — CI reverts them.
- **Frozen gates:** `gates.toml` and `gates.lock` are not yours to edit. A failing gate is reported,
  never silenced.
- **Commits:** clear, scoped messages; commit `Cargo.lock`; keep `cargo fmt --check` and `cargo clippy`
  clean. Do **not** put any AI/model identifier (e.g. "Claude", model names) into commit messages, code
  comments, or any committed file.
- **Scope discipline:** touch only the crates your task card assigns; another agent may own the rest.
- **When blocked or a number feels like a guess:** escalate to the lead reviewer with the specifics —
  never invent a plausible value.

Detailed maps: [`implementation/VERIFICATION.md`](implementation/VERIFICATION.md) (why green must mean
working), [`implementation/tests/`](implementation/tests/README.md) (per-phase task cards),
[`implementation/METRICS-AND-GATES.md`](implementation/METRICS-AND-GATES.md) (exact thresholds).
