# drone-sim

Phase 1 simulation for the ENGR 4530 vision-guided person-following drone (BetaFPV Meteor65 Pro, shared autonomy). It models the Betaflight-stabilized drone as the ground station sees it, the camera keypoint sensor, the command link, and the yaw and range control loops.

`SPEC.md` is the source of truth for models, parameters, scenarios, figures, and build order.

## Commands

```
pip install -e ".[dev]"
pytest
python scripts/tune_gains.py
python scripts/run_scenarios.py
python scripts/sweep.py
python scripts/make_figures.py
```

## Working Rules

- Follow `SPEC.md`. Ask before adding models, features, scenarios, or dependencies it does not list.
- Make only the change requested. Propose a plan and wait for confirmation before multi-file changes.
- Run `pytest` before reporting a task done.
- Work in the build order in SPEC Section 11. Finish Must figures before Should figures.

## Code Conventions

- SI units internally: meters, seconds, radians. Degrees appear only in config keys ending `_deg` and on plot axes.
- Sign conventions follow SPEC Section 3.1. Tests enforce them; do not change a sign to make a test pass without asking.
- `controllers.py` and `estimator.py` import only numpy and the standard library. The ground station will import them directly.
- All parameters come from `config/`. No numeric constants in code beyond physical constants and unit conversions.
- Seed every random generator and record the seed in saved results.
- Figures come only from `scripts/make_figures.py`. Never edit figure files by hand.
- Python 3.10 or later. Dependencies: numpy, scipy, matplotlib, python-control, pyyaml, pytest.

## Writing (Report and Documentation)

- Active voice, US spelling, no em dashes.
- Define every technical term on first use. No internal shorthand.
- One term per concept: range, bearing, pitch command, yaw-rate command, total delay, effective delay.
- Write the report as the report itself, not as narration about the work.
