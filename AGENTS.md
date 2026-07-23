# AGENTS.md — operating rules for any agent working in this repo

You are contributing to **CCR** (Convergent Concept Representations), a from-scratch research
codebase whose entire value depends on **trustworthy numbers**. Read this before writing code. It is
short on purpose; the detailed specs are linked. These rules are **binding**, not advisory — a change
that violates one is wrong even if it compiles and even if `cargo test` is green.

The governing law of this repo, in one sentence: **green tests do not mean the code works; a test is
only evidence if it has been shown to fail on broken code.** The full architecture is
[`implementation/VERIFICATION.md`](implementation/VERIFICATION.md); your task cards are in
[`implementation/tests/`](implementation/tests/README.md); the frozen numeric thresholds are
[`gates.toml`](gates.toml) (defined in [`implementation/METRICS-AND-GATES.md`](implementation/METRICS-AND-GATES.md)).

---

## The scar that created these rules (read it — do not repeat it)

An agent was asked to build the synthetic-data fixture whose entropy is the "known answer" a model
must reach. It shipped a **green** test suite. It was broken two ways: (1) it generated **uniform
random bytes** — there was no Markov source at all; (2) it computed the "true" entropy from the
**generated sample's own counts** and then compared that to an "empirical" entropy computed from **the
same counts** — a number compared to itself, so the difference was `0.000000`. The test validated
nothing; had it merged, every downstream model would have been judged against a meaningless target.

Three rules come directly from that scar. They are the first three below because they are the ones most
easily violated while everything looks green.

---

## The hard rules

1. **A metric/known-answer test's expected value MUST be computed independently of the thing under
   test.** Never derive the target from the same sample, output, or code you are validating. The truth
   for a data fixture comes from the parameters that *define* the source (e.g. the transition matrix),
   not from counting what you generated. If the "expected" and "actual" share a computation, the test
   is vacuous — delete it and start over.

2. **Every known-answer test (`kat_*`, `kae_*`) MUST ship a paired anti-vacuity canary** — a test that
   feeds a *wrong* input and asserts the check *fails* (e.g. `test_entropy_test_can_say_no`). A metric
   that has never been shown to say "no" is not evidence. CI enforces this (`anti-vacuity-lint`).

3. **Treat an exact-zero difference between two supposedly independent quantities as a BUG, not a
   success.** `|a − b| == 0.000000` between an analytic and an empirical estimate almost always means
   you compared something to itself. Investigate before celebrating.

4. **Test-first: write the test and watch it FAIL on the empty stub before implementing.** A test that
   is green before you write the code tests nothing — rewrite it.

5. **Never type, predict, or transcribe a measured number into a Markdown/text file.** Numbers exist
   only in `runs/*/metrics.json` and reach docs only through the generated `results/RESULTS.md`. Cite
   "see RESULTS.md"; do not paste the digits. CI (`verify-results`) reverts hand-edited numbers.

6. **Never weaken a gate to pass.** Do not edit `gates.toml`/`gates.lock`, do not loosen a tolerance,
   do not delete or `#[ignore]` a failing test or a negative control. A failing gate is a **finding**
   to report, which is the entire purpose of the gate. If you believe a threshold is wrong, say so in
   your report — do not change it.

7. **Report the truth, including red.** A red test honestly reported is worth far more than a green one
   you cannot explain. Never claim a number you did not produce from an actual run. If you could not
   finish something, say so and why.

8. **Determinism (C4):** seed every RNG (`rand_chacha`/seeded), CPU is the source of truth, same seed →
   bit-identical output. If a "reproducible" run is not reproducible, that is a bug to fix, not ignore.

9. **Stay in your assigned scope.** Touch only the crates/files your task card names. Do not "helpfully"
   refactor another agent's crate; it causes merge conflicts and hides regressions.

10. **If a number feels like a judgment call, your task card is under-specified — escalate, do not
    invent.** Ask the lead (report the ambiguity); never paper over it with a plausible-looking value.

11. **Reuse the reference, do not reinvent (and do not depend on it).** Before implementing a standard
    component, read its row in [`implementation/PRIOR-ART-AND-REUSE.md`](implementation/PRIOR-ART-AND-REUSE.md);
    port the algorithm, the default hyperparameters, and — where possible — the reference's **test
    vectors** (VICReg loss on a fixed batch, ECE matching torchmetrics, bind/unbind properties from
    torchhd). Do not re-derive a standard algorithm from scratch, and do not add a framework dependency
    to get it. The novelty here is the *intersection and the engineering*, not the primitives.

12. **Minimal dependencies — a reviewer must survive `cargo tree`.** The entire core budget is
    `rand`+`rand_chacha`, `serde`+`serde_json`, `sha2`, `toml`, and `proptest`/`approx`/`tempfile`
    (dev-only). Any new crate must be justified in the commit against that budget. **No DL framework**
    (`burn`/`candle`/`tch`/`dfdx`) for headline numbers without lead sign-off; prefer a small hand-rolled
    implementation (e.g. the ~400-line autodiff tape for Phase 1) over a heavy dependency. Prefer a
    hand-rolled `O(d²)` DFT for tests over pulling an FFT crate. Fewer deps = fewer pins =
    reproducibility (C4) and no dependency hell.

13. **Commit small, push often — a human reviews incrementally.** Work in small, logically-scoped
    commits (one task card ≈ one commit), and **push your branch to `origin` after each commit or two**
    (`git push -u origin <your-branch>`) so a human reviewer can follow the work as it lands, not in one
    big-bang merge at the end. Do not sit on a large uncommitted/​unpushed diff. Keep each pushed commit
    green (`cargo test` passing, `fmt`/`clippy` clean) so every review point is a working state. Prefer
    porting a cited reference (rule 11) over re-deriving — a solved problem should cost review time, not
    build time.

---

## Mandatory pre-report self-review (answer every line before you say "done")

Copy this into your final report with each box checked or explicitly explained:

- [ ] Every metric/known-answer test's expected value is computed **independently** of the code under test (Rule 1).
- [ ] Every `kat_*`/`kae_*` has a **paired canary** proving it can fail on a wrong input (Rule 2).
- [ ] No "independent" comparison yields an **exact-zero** difference; if one does, I explain why it is legitimate (Rule 3).
- [ ] I watched each test **fail on the stub** before implementing (Rule 4).
- [ ] I typed **no measured numbers** into any Markdown file; results come from `report` (Rule 5).
- [ ] I did **not** weaken any threshold, tolerance, or test to pass; `gates.toml` is untouched (Rule 6).
- [ ] Same seed → **bit-identical** output (Rule 8).
- [ ] I consulted the **reference** for each standard component and ported its test vectors where possible (Rule 11).
- [ ] I added **no new dependency** beyond the justified budget; no DL framework without sign-off (Rule 12).
- [ ] I report the **real** `cargo test` summary and every RED/incomplete item, with reasons (Rule 7).

An agent that reports "done" without this checklist has not finished.

---

## What "done" means

A unit of work is done only when its task card's Definition-of-Done (see the phase file in
`implementation/tests/`) is met: unit + property tests green **and** the negative controls fail on the
broken variants as designed **and** the catalog mutants are killed **and** the known-answer end-to-end
reaches its independently-computed target **and** the gate PASSes from a provenance-checked artifact.
`cargo test` being green is the floor, not the finish line.

## How to report back to the lead

Branch name + final commit SHA; the exact `cargo test` summary (pass/fail counts, not a paraphrase);
which DoD bullets are GREEN vs RED; the pre-report checklist above; every assumption you made; and
anything you could not finish and why. Truthful and specific beats green and vague.
