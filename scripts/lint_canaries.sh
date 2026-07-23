#!/usr/bin/env bash
# anti-vacuity-lint — enforce AGENTS.md rule 2 / VERIFICATION.md §7.
#
# A test needs a paired anti-vacuity canary (a sibling test proving the check can FAIL on a
# wrong input) when it is self-reference-prone: either
#   (a) an end-to-end known-answer test  (file named  kae_*.rs), or
#   (b) any test file that compares a COMPUTED TRUTH against an INDEPENDENT ESTIMATE
#       (contains an analytic/expected/truth token AND an empirical/sample/measured token).
# Pure algebraic-identity KATs (Sum p = 1, ||U psi|| = ||psi||, U-dagger U = I) are exempt —
# their "can it fail" proof is the mutation catalog, not a canary.
#
# Why this exists: a real defect shipped a green entropy test that computed the "truth" from
# the generated sample's own counts and compared it to an empirical entropy from the same
# counts (|delta| = 0.000000) — a number compared to itself. A canary makes that impossible
# to ship green.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

canary_marker='can_say_no|canary|vacuit|_can_fail|rejects_wrong'
truth_marker='analytic|expected|ground_truth|_truth|reference'
estimate_marker='empirical|sample|estimate|measured'

mapfile -t testfiles < <(git ls-files '**/tests/*.rs' 'qilm-*/tests/*.rs' 2>/dev/null | sort -u)

fail=0
checked=0
for f in "${testfiles[@]}"; do
  base="$(basename "$f")"
  needs=0
  case "$base" in
    kae_*.rs) needs=1 ;;
  esac
  if [ "$needs" -eq 0 ]; then
    if grep -Eqi "$truth_marker" "$f" && grep -Eqi "$estimate_marker" "$f"; then
      needs=1
    fi
  fi
  [ "$needs" -eq 0 ] && continue

  checked=$((checked+1))
  if grep -Eq "$canary_marker" "$f"; then
    echo "  OK   $f (canary present)"
  else
    echo "  FAIL $f — needs an anti-vacuity canary (a *can_say_no/*canary test proving it can fail)"
    fail=1
  fi
done

if [ "$checked" -eq 0 ]; then
  echo "anti-vacuity-lint: no self-reference-prone known-answer tests yet — OK."
  exit 0
fi
if [ "$fail" -ne 0 ]; then
  echo; echo "anti-vacuity-lint FAILED (AGENTS.md rule 2). Add a canary or make the truth independent of the sample."
  exit 1
fi
echo "anti-vacuity-lint: all $checked self-reference-prone test file(s) have canaries."
