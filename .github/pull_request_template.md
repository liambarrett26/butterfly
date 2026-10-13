## What and why

Closes #

<!-- What changed and why. One roadmap item per PR. -->

## Tests

- [ ] `python tests/run_synthetic.py` exits 0 (N/N checks)
- [ ] New behaviour has a new check; a bug fix has a check that fails without the fix
- [ ] CI is green

## Effect on picks

<!-- For anything that changes picks: a before/after `--dry-run` on a real folder,
     as a plan-table diff or a description of the report timelines.
     Otherwise write "None". -->

## CACHE_VERSION

<!-- Bumped to N, because …  /  Not bumped: analysis output unchanged. -->

## Checklist

- [ ] README: roadmap item ticked, and a line under `Unreleased` in the changelog
- [ ] Everything runs locally: no uploads, no cloud APIs
- [ ] `git status` shows no footage, caches, outputs or model weights staged
