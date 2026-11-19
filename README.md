# FRE-GY-9743-Assignments-2026fall

## HW8 — implementation notes

The five TODO blocks are implemented: the OIS annuity and par rate (2a, 2b), the
par-rate gradient (2c), the calibration Jacobian (3) and the conversion of state
risk to market-quote risk (4). The notebook runs end to end and reproduces the
supplied outputs (PV −24,829.03; risk and hedge tables to the printed precision).
An appendix at the end of the notebook, marked as not part of the template, adds
a curve plot, both views of the risk and a re-calibrated hedge check.
Documentation lives in [`docs/`](docs/):

- [docs/hw8-implementation.md](docs/hw8-implementation.md) — the economics (OIS,
  par rate, annuity, why quote risk), each task's maths, code and design
  reasons, the results explained, the verification, and points flagged in the
  supplied code.
- [docs/calibration-and-risk-architecture.md](docs/calibration-and-risk-architecture.md)
  — the path from quotes to a hedge, who owns what, why the bootstrap is
  sequential, and a recipe for adding a calibration instrument.
- [docs/code-reuse.md](docs/code-reuse.md) — relationship with Assignments 1–7:
  nothing copied; what changed in the shipped library and how earlier behaviour
  was re-checked.
- [docs/session-report.md](docs/session-report.md) — what was added, modified
  and is new, with the mutation-test results.

Theoretical background: [`docs/stochastic_calculus_and_yield_curve_calibration.md`](../../docs/stochastic_calculus_and_yield_curve_calibration.md).

Requirements: the notebook reads the quotes with `openpyxl` (pinned in
`requirements.txt`). Run the regression tests (about two minutes) with:

```bash
python tests/test_ois_calibration_and_risk.py
```
