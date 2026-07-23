#!/usr/bin/env bash
# anti-vacuity-lint — enforce AGENTS.md rule 2 / VERIFICATION.md §7:
# every known-answer test file (tests/kat_*.rs, tests/kae_*.rs) MUST contain a paired
# anti-vacuity canary (a test proving the check can FAIL on a wrong input). A known-answer
# test with no canary is not evidence. Exit non-zero listing any offenders.
#
# Rationale: a real defect shipped a green entropy test that compared a number to itself.
# A canary (feed a wrong source, assert the check fails) is the cheapest structural guard.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# git-tracked known-answer test files only
mapfile -t files < <(git ls-files '**/tests/kat_*.rs' '**/tests/kae_*.rs' 'qilm-*/tests/kat_*.rs' 'qilm-*/tests/kae_*.rs' 2>/dev/null | sort -u)

if [ "${#files[@]}" -eq 0 ]; then
  echo "anti-vacuity-lint: no known-answer test files yet — OK (nothing to enforce)."
  exit 0
fi

marker='can_say_no|canary|vacuit|_can_fail|rejects_wrong'
fail=0
for f in "${files[@]}"; do
  if grep -Eq "$marker" "$f"; then
    echo "  OK   $f (canary present)"
  else
    echo "  FAIL $f — no anti-vacuity canary (needs a test named *can_say_no/*canary proving it can fail)"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "anti-vacuity-lint FAILED: every kat_*/kae_* known-answer test must ship a canary (AGENTS.md rule 2)."
  exit 1
fi
echo "anti-vacuity-lint: all ${#files[@]} known-answer test file(s) have canaries."
