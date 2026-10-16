"""Run every scenario once without noise and once per seed with noise.

Writes results/<scenario>_clean.npz and results/<scenario>_seed<NN>.npz.
Each file records its seed in its metadata.
"""

from __future__ import annotations

from pathlib import Path

from dronesim.params import load_gains, load_params
from dronesim.scenarios import build_scenarios
from dronesim.sim import run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    params = load_params()
    gains = load_gains()
    mc = params.monte_carlo
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, scenario in build_scenarios(params).items():
        run(scenario, params, gains, seed=mc.base_seed, noise=False).save(
            RESULTS_DIR / f"{name}_clean.npz"
        )
        for i in range(mc.n_seeds):
            seed = mc.base_seed + i
            run(scenario, params, gains, seed=seed).save(RESULTS_DIR / f"{name}_seed{seed:02d}.npz")
        print(f"{name}: 1 noise-free run and {mc.n_seeds} noisy runs saved")


if __name__ == "__main__":
    main()
