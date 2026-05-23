# Flocking for Traffic — investigation code & writeup

Empirical study of the McKenzie / Olfati-Saber flocking algorithm
(α + β + γ + τ control law) applied to vehicle traffic. The repo
contains the experiments that delimit the algorithm's working envelope,
three proposed algorithmic extensions, and a four-experiment evaluation
against traffic-theory benchmarks.

See **[writeup.md](writeup.md)** for the formal write-up and
**[investigations_queue.md](investigations_queue.md)** for the research log.

## Layout

```
flocking_lib/             core algorithm modules
  sigma_norm.py, rho.py, phi*.py
  a_ij.py, n_ij.py, neighbour.py
  control_alpha.py, control_beta*.py, control_gamma.py, control_tau.py
  dynamics.py, flock_layouts.py, multi_flock_sim.py

experiments/              all runnable experiment scripts
  exp_fundamental_diagram.py     Exp A · max stable density
  exp_smoothness.py              Exp B · smoothness vs lane-locked
  exp_capacity.py                Exp C · lane-less vs lane-based perturbation
  exp_intersection_mfd.py        Exp D · intersection MFD
  diagnose_*.py                  failure-mode diagnostics
  exp_geometries.py, exp_intersection.py, exp_cross_*.py, exp_merge*.py
  exp_targeted_gamma.py, exp_predict_suppress.py, exp_rotation_matrix.py
  exp_unified.py, test_tau_variants.py
  sweep_*.py                     parameter sweeps
  _bootstrap.py                  path helper (imported by every script)

figures/                  generated PNG plots
animations/               generated GIF visualisations
legacy/                   older drafts (ivt_projectver2.py, flocking_main.py, flocking_explorer.html)

writeup.md                formal writeup (intro / methodology / results / discussion)
investigations_queue.md   research log with full per-investigation findings
```

## Running an experiment

From the project root:

```sh
python experiments/exp_fundamental_diagram.py
python experiments/exp_smoothness.py
python experiments/exp_capacity.py
python experiments/exp_intersection_mfd.py
```

Each script also works from inside `experiments/`. The tiny
`_bootstrap.py` module (imported once at the top of each script) makes
`from flocking_lib.X import Y` resolve regardless of cwd. Output
PNGs / GIFs are written to the current working directory; move them
into `figures/` or `animations/` if you want to keep them.

## Dependencies

`numpy`, `matplotlib`. Tested on CPython 3.11+.

## Key findings (one-paragraph summary)

The McKenzie algorithm is a steady-state cruise controller, not a
traffic-flow model: in a translation-invariant corridor it pins mean
forward speed to v_d at every density, so the classical q-k-v
fundamental diagram is degenerate. Reframed as max-stable-density,
the algorithm achieves 1.18× the HCM road-width-equivalent reference
at lattice integrity and 1.35× at no-overlap. Lateral freedom gives
the algorithm a 3-26× smoothness advantage below lattice saturation
and prevents rear-end near-collisions during incidents (intra-min
≥ 1.5 m vs 0.03 m for a lane-based baseline at high density). The
V2 extension (R(+90°) τ, predictive suppression, target γ) unlocks
one-shot intersection geometries but cannot sustain continuous
demand: each τ tuning has exactly one "Goldilocks" density at which
a rotational pattern self-organises. The lane-less paradigm pays
off as an incident-resilience and smoothness layer, not as a
replacement for conventional traffic engineering.
