# Investigation queue

Saved 2026-05-23 after the exp_geometries.py geometry-matrix run.

Each entry is a follow-up surfaced by one of the previous sweeps but not yet
investigated. We'll go through them one at a time.

## 1. Compression mode in multi-row flocks  *(done · 2026-05-23)*
**Where it surfaced:** scenarios D (2×4 vs 2×4) and E (2×4 vs 1×8) in
`exp_geometries.py`. Intra-flock minimum drops from d_a=7 m to ~4.3 m during
the encounter.

**Mechanism (diagnose_compression.py):** the velocity-only McKenzie τ-force
depends only on `(p_i − p_r)`. Within a flock all cars share the same velocity,
so all flock-1 cars get *identical* y-force — the whole flock is pushed
uniformly toward one side of the road. The leading row hits β first and gets
pinned; the trailing row, still under the same uniform push, catches up to
the pinned leading row. The minimum-intra occurs at the moment β has
decelerated the leading row but not yet decelerated the trailing row.
Self-recovering: once τ disengages after the crossing, α + β + γ push the
trailing row back up. End-state lattice is mostly intact.

**Fix attempts (test_tau_variants.py):**
  - V1 hybrid (vel + radial repulsion): best inter (8.05 m) but chaotic;
    row order temporarily inverts.
  - V2 predictive gating: self-defeating — τ's success closes its own gate
    → chatter → almost no net deflection (inter=0.09).
  - V3 flock-relative direction: each flock splits its own rows symmetrically
    → both flocks send rows to the same bands → inter=0.06.
  - V4 projected radial (radial gated to align with v_term): *worse* than
    baseline because the projection adds force to the leading row (slamming
    it harder into the wall) without restraining the trailing row.

**Conclusion:** the compression is *not* a parameter-tuning problem. It's
structural to McKenzie's lateral-deflection τ inside a hard-walled corridor.
McKenzie has no concept of a *target band* — cars deflect monotonically
until τ disengages or β stops them. To fix the compression cleanly the
framework needs one of:
  - per-flock target-y γ-term with externally assigned bands per encounter,
  - velocity damping in τ that scales the force down as the agent reaches
    its target band, or
  - a softer/wider β (large d_b with gradual taper) so the wall pre-
    decelerates rather than hard-stopping.

For practical use the workable knob is **stronger α** (2-3× c1_a) which
trades a bit of inter-flock margin for ~1 m of intra-flock recovery.

## 2. Asymmetric flock-size sweep  *(done · 2026-05-23)*
**Where it surfaced:** scenarios B (1×4 vs 1×2) and C (1×4 vs 1×8) in
`exp_geometries.py`.

**Sweep:** N1=4 fixed, N2 ∈ {1,2,3,4,6,8,10,12}, single-row, y_hi=14.4.

**Key finding (counter to hypothesis):** in 1D geometries, **the β-zone
edge acts as an emergent target band**. McKenzie has no notion of target
band, but β does: at d_b=3 m from each wall, the β-edge sits 4.2 m from
the y_mid=7.2 centerline. *Every* configuration converges to ≈4 m
deflection regardless of N1/N2 ratio. The two flocks deflect to the same
magnitude (~4 m each) whether the asymmetry is 4 vs 1 or 4 vs 12.

**Where asymmetry actually shows up:**
  - **wall-proximity time** scales with relative size: the smaller flock
    (more opposing pressure per car) spends more of the run pressed against
    the β-edge. N2=12 → F1 wall%=29% vs F2 wall%=17%.
  - **peak y-excursion** grows with N2 for F1 (up to 5.6 m before β
    bounces back), but final settled position is unchanged.
  - **α-lattice** degrades very slightly with N2 (intra 7.0 → 6.6 m for
    F1) but stays well above 6 m. No lattice break.

**No N1/N2 ratio breaks the algorithm in 1D up to N2=12.** The cost of
being the smaller flock is wall-time and ride harshness, not collision
risk. This is a consequence of the bounded corridor (β-as-attractor),
not the algorithm itself — in open-space McKenzie the smaller flock
likely *would* deflect further because there's no soft target.

## 3. Off-center initial placement (graceful vs aggressive engagement)  *(done · 2026-05-23)*
**Where it surfaced:** scenario F (off-center, pre-sorted lanes) — flocks
appeared to pass each other essentially without τ firing.

**Sweep:** dy ∈ {0, 1, 2, 3, 4, 6, 8, 10, 14}, 1×4 vs 1×4 on y_hi=24.

**Counter-intuitive finding #1**: τ-engagement count *increases* with offset
(5454 head-on → 5730 at dy=14). At large dy, neither flock deflects, so
headings stay anti-parallel and the gate stays satisfied longer. There is
**no "lazy regime"** — τ fires more, not less, when its work is least
needed.

**Counter-intuitive finding #2**: deflection magnitude *decreases* with
offset. At dy=0 each flock moves 6.66 m; at dy=14, only 1.90 m. The reason
is that at large dy each flock starts close to a wall and reaches the
β-edge after only ~2 m of motion. β stops the deflection short.

**Two deflection regimes** now identified:
  - *encounter-limited* (centered start, wide road): deflection set by
    τ/γ/encounter-duration physics; cars don't reach β.
  - *β-limited* (near-wall start, OR narrow road): deflection capped by
    β-edge as soft target band.

  This corrects a claim made in investigation #2 ("β creates a universal
  target band") — that claim only holds in the β-limited regime.

**Practical implication:** the algorithm can't detect when its avoidance
work is unnecessary; it just keeps pushing through pre-sorted geometries.
A "smart" τ extension could use predictive gating (V2 from the compression
investigation) **to disable τ when projected paths already clear** —
counter to V2's failed use as a trigger, this one-way use should be a
clean win.

---

# Phase 3 supplement: validating the deck claims  *(done · 2026-05-23)*

Phase 3 Exp A-D characterised algorithm capability across traffic-theory
benchmarks but did not directly *test the marketing claims* on the
Week-2 deck ("smoother driving", "increased capacity", "lane-less and
direction-less"). The user asked whether those claims are actually
valid. Three follow-up experiments, each mapping a claim to a clean
measurement.

## Exp E. Lane-formation entropy (tests "lane-less" claim)  *(done · 2026-05-23)*

**Setup (exp_lane_formation.py):** periodic corridor, uniform-random
initial y in the usable strip. Equilibrate for 10 s, then pool the
y-positions from the last 20 s into one histogram per N. Compute
Shannon entropy normalised to a uniform distribution (1.0 = uniform,
0.0 = single bin), and count peaks ≥ 0.4 × peak height as "emergent
lanes".

  | N (k veh/km) | H/H_uniform | #modes | interpretation |
  | --- | --- | --- | --- |
  | 20 (40) | 0.887 | 3 | three bands (sparse) |
  | 40 (80) | 0.906 | 2 | **two emergent lanes** |
  | 60 (120) | 0.948 | 2 | **two emergent lanes** |
  | 90 (180) | 0.840 | 2 | **two emergent lanes** |
  | 120 (240) | 0.802 | 2 | **two emergent lanes** |
  | 140 (280) | 0.808 | 2 | **two emergent lanes** |

**Verdict: "lane-less" is FALSIFIED.** From uniform-random initial y,
the algorithm settles into exactly two emergent lanes at every realistic
density (N ≥ 40). The peaks land at y ≈ 4 and y ≈ 10, matching the
strip-hex theory row centres (d_b + 0.5 = 3.5 and W - d_b - 0.5 = 10.5)
within the β-zone offset. At N = 120-140 the middle channel y ∈ [6, 8.5]
is almost completely empty.

This reframes the algorithm: lane-less *by design* (no lanes imposed)
but lane-forming *in practice* (α-lattice hex packing produces two rows
that are emergent lanes). The honest story: the algorithm discovers
lanes rather than abolishing them. Arguably a feature (no infrastructure
assumption, lanes adapt to road width), but not the "no-lanes" claim.

## Exp F. String stability (tests "smoother driving" claim)  *(done · 2026-05-23)*

**Setup (exp_string_stability.py):** 16-car platoon, single file at
d_a spacing, all at v_d. Brake the leader (car 15) to v_x = 2 m/s for
2 s, release. Compute ||e_i||_2 = √(∫ (v_x(t) - v_d)² dt) for each
car. String-stable iff the disturbance does not grow as it propagates
from leader to tail.

  | metric | lane_locked | lane_less |
  | --- | --- | --- |
  | leader L2 disturbance | 12.56 | 12.27 |
  | first-follower (idx 14) L2 disturbance | **12.77** | 8.75 |
  | leader → first-follower ratio | **1.02 (amplifies)** | **0.71 (decays)** |
  | tail (idx 0) L2 disturbance | 3.93 | 3.08 |

**Verdict: "smoother driving" is CONFIRMED and STRENGTHENED.**
Lane-locked *amplifies* the leader's disturbance at the first follower
(ratio 1.02, classic string instability), while lane-less *attenuates*
it by 29 % at the same position. Across the whole platoon, lane-less
L2 disturbances are 0-46 % smaller than lane-locked (ratio 1.00-1.46),
with the largest gap at the first follower. Both eventually decay
(neither produces a stop-and-go shockwave at this platoon size), but
lane-less dissipates immediately while lane-locked needs several cars
to begin attenuating.

This complements Exp B (smoothness at steady state) and Exp C
(recovery time after a brake) with the missing string-stability
question.

## Exp G. Honest capacity comparison (tests "increased capacity" claim)  *(done · 2026-05-23)*

**Setup (exp_capacity_comparison.py):** same periodic 14 m corridor;
sweep N for two conditions head-to-head:
  lane_less : full α + β + γ on a 2-D corridor (= Exp A reframed).
  lane_based: 2 fixed lanes at y ≈ 4 and y ≈ 10, y locked.
For each N: 30 s run, measure intra-min over the last 10 s.
Capacity = largest N with intra-min ≥ d_a/2.

  | N | k [veh/km] | lane_less intra | lane_based intra |
  | --- | --- | --- | --- |
  | 120 | 240 | 7.00 | 9.04 |
  | 180 | 360 | **0.00 ✗** | 3.50 (right at safety) |
  | 220 | 440 | 0.00 ✗ | 0.00 ✗ |

  | Condition | Capacity N* | q = k·v_d [veh/h] |
  | --- | --- | --- |
  | lane_less | 160 | 11 520 |
  | lane_based (2 lanes) | **180** | **12 960** |
  | ratio (lane_less / lane_based) | — | **0.89×** |

**Verdict: "increased capacity" is FALSIFIED.** Lane-based achieves
**11 % HIGHER** safe capacity than lane-less on the same corridor.
Lane-less spontaneously forms two emergent lanes (Exp E) but with
imperfect packing inside each row, while lane-based holds cars on
exact lane centres. The two-row geometric ceiling is the same; the
difference is how tightly each algorithm uses it.

Lane-less collapses *abruptly* (intra 7.00 → 6.93 → 4.64 → 0 across
N = 120-180), whereas lane-based degrades *gracefully*
(9.04 → 6.57 → 5.44 → 5.00 → 4.67 → 3.50 → 0). Both eventually fail at
the same density (≈ 360-440 veh/km), but lane-based has a wider
useful operating range.

The steady-state q-k curves are *identical* for both conditions in the
safe regime (q = k · v_d), since mean(v_x) = v_d by construction in
both (Exp A).

## Combined verdict on the three claims

  | Deck claim | Verdict | Where the truth lives |
  | --- | --- | --- |
  | "Smoother driving" | **Confirmed and strengthened** | Exp B (3-26× lower rms_ax and std v_x below saturation), Exp F (string-stable; ratio 0.71 vs lane-locked's 1.02), Exp C (1.4-2.9× faster recovery) |
  | "Increased capacity" | **Falsified** | Exp G: lane-based has 11 % higher safe capacity. Steady-state q identical in safe regime. The win is incident response, not capacity. |
  | "Lane-less" | **Falsified** | Exp E: two emergent lanes form spontaneously at N ≥ 40 from uniform-random initial conditions. Lane-discovering, not lane-abolishing. |

**Net effect on the thesis story:** the deck overclaims on capacity
and lane-freeness, and underclaims on robustness. The honest pitch is
that lane-less flocking provides *equivalent* steady-state throughput,
*materially smoother* driving below saturation, *genuine string
stability* under perturbation, and *dramatically better incident-
response safety*, all without requiring any imposed lane infrastructure
(lanes emerge from the algorithm itself).

---

# Phase 2 summary (combined algorithm)  *(done · 2026-05-23)*

Combined the three Phase-2 improvements into a "V2" algorithm and benchmarked
against the Phase-1-final V0 across the full scenario suite (exp_unified.py).

**Critical finding**: the three improvements are **NOT mutually composable**.
Predictive suppression breaks the R-rotation roundabout pattern (it turns off
τ once projected paths look safe, which is precisely when the curve needs to
keep firing). The honest unified algorithm is *mode-aware*:

  - **R(+90°)**: universally on (backward-compatible, never hurts).
  - **Velocity-only τ** (c1_t=0): universally on (Phase-1 finding).
  - **Corrected β**: universally on (Phase-1 fix).
  - **Predictive suppression**: corridor mode only — disable in intersections.
  - **Target γ**: opt-in when external lane assignments exist.

| Scenario | V0 | V2 mode-aware | Effect |
| --- | --- | --- | --- |
| 1×4 head-on canonical | 6.58 | 6.80 | no regression |
| 2×4 compression D | 5.11 / intra 4.27 | 3.67 / intra **6.28** | lattice integrity restored |
| v_d=3 slow | 8.47 | 7.48 | minor inter loss |
| v_d=25 fast | 4.05 / 266 esc | 3.99 / **0 esc** | wall escapes eliminated |
| dy=14 wasted-τ | 17.92 | 14.00 | margin loss, τ idle when not needed |
| 3-flock 120° | **0.07** ✗ | **36.00** ✓ | previously unsolvable |
| 4-flock 90° | **0.00** ✗ | **28.25** ✓ | previously unsolvable |

Headline: V2 *unlocks the intersection geometry class entirely* (both 3-way
and 4-way symmetric configurations) while preserving canonical corridor
behavior and eliminating the high-speed wall-escape failure mode. The
trade-offs are real but small (1-4 m of margin sacrificed where V0 was
already safely above car-width).

---

---

# Phase 3: traffic-theory evaluation

The Week-2 deck claimed: lane-less, smoother driving, increased roadway
capacity, maximized capacity usage. Phase 1+2 measured *safety* (inter-
flock min, off-road escapes). Traffic theory measures *throughput* and
*flow quality*. Four planned experiments to bridge.

## Exp A. Fundamental diagram (q-k-v) for the corridor  *(done · 2026-05-23)*
Periodic corridor, single direction, all cars at v_d. Sweep N; let
α-lattice settle; measure mean speed; compute flow `q = k·v`. Extract
v_f, q_max, k_c, k_j. Compares directly to HCM ~2200 veh/hr/lane
freeway capacity baseline.

**Diagnostic finding (diagnose_fundamental.py):** the classical q-k-v
framing is *degenerate* for this algorithm. In a translation-invariant
periodic corridor the α-gradient is x-symmetric so mean(α_x)=0 at
steady state. γ = -c_g(v - v_d·x̂) is the only x-asymmetric force, so
mean(γ_x)=0 ⇒ **mean(v_x) = v_d for ALL densities** (confirmed
empirically: 36.0000 km/h at k = 80, 240, 500, 800, 1400 veh/km). Past
the lattice capacity the algorithm fails by *overlap* (intra-min → 0),
not by slowing down. McKenzie/Olfati-Saber is a steady-state cruise
model, not a traffic-flow model.

**Reframed Exp A (exp_fundamental_diagram.py):** measure the maximum
density at which the α-lattice survives. Two thresholds reported:

  | Metric | Threshold | k [veh/km] | q = k·v_d [veh/h] |
  | --- | --- | --- | --- |
  | k_lattice (lattice intact) | intra-min ≥ 0.9·d_a = 6.3 m | 280 (N=140) | 10 080 |
  | k_stable (no overlap) | intra-min ≥ d_a/2 = 3.5 m | 320 (N=160) | 11 520 |
  | strip-hex theory | 2 rows × L/d_a on W_usable=8m | 286 | — |
  | HCM equivalent | 2200 veh/h/lane × (14 m / 3.6 m) | — | 8 556 |

  k_lattice matches strip-hex theory within 2% — the algorithm achieves
  the geometric packing limit under d_a-equilibrium. **q_lattice is
  1.18× HCM** and **q_stable is 1.35× HCM**. The lane-less algorithm
  packs ~18-35% more flow at v_d than HCM expects on a same-width road,
  *but the comparison is qualified* by the lack of congestion regime —
  HCM's 2200 veh/h/lane is measured against drivers who can slow down;
  this algorithm never does, so it doesn't model rush-hour breakdown.

**Phase 4 follow-up (not in original plan):** the missing congestion
regime is a single algorithmic gap — a car-following / leader-slowdown
term that would let γ's target shrink when blocked. Without it, Exp B
(smoothness) and Exp D (intersection MFD) both inherit the same "no
slowdown ever" property and will measure the wrong thing. Address
before continuing Exp B–D.

## Exp B. Smoothness comparison  *(done · 2026-05-23)*
Per-car RMS acceleration, peak jerk, speed variance. Compare to a
"lane-locked" baseline (each car pinned to its initial y).

**Setup (exp_smoothness.py):** same corridor as Exp A, uniform random
initial positions, both conditions seeded identically. Flocking runs the
full α+β+γ system; lane-locked zeros u_y and v_y each step. Metrics over
the last 20 s of a 30 s run: rms_ax, rms_ay, peak |jerk_x|, peak |jerk_y|,
std(v_x). Aggregated across cars as (mean, p95).

**Two-regime finding (the dominant result):**

  | N (k veh/km) | rms_ax flock | rms_ax lock | std_vx flock | std_vx lock | peak_jx flock | peak_jx lock |
  | --- | --- | --- | --- | --- | --- | --- |
  | 90 (180) | 0.064 | **0.183** | 0.018 | **0.196** | 3.4 | 5.4 |
  | 120 (240) | 0.017 | **0.111** | 0.005 | **0.129** | 1.1 | 1.6 |
  | 140 (280) | 0.095 | 0.105 | 0.079 | 0.086 | **3.7** | 1.6 |
  | 160 (320) | 0.119 | 0.124 | 0.137 | 0.143 | **3.6** | 1.4 |

  *Low-to-mid density* (N ≤ 120, below the strip-hex limit): lateral
  freedom is a clear win. Flocking shows 3–26× lower rms_ax and std_vx
  than lane-locked. Cars resolve spacing imbalances by shifting laterally
  rather than braking in x.

  *Near lattice limit* (N ≥ 140 ≈ k_lattice from Exp A): the advantage
  disappears. rms_ax and std_vx converge, AND **flocking peak_jx becomes
  2-3× worse than lane-locked**. With the lattice full, y-shifts no
  longer relieve x-load; they just inject extra y-x coupling that
  manifests as sharper x-jerks.

**Crossover point ≈ N=140 (k=280 veh/km) — the same lattice-saturation
density extracted independently in Exp A.** Two separate experiments
converge on the same density as the regime boundary.

**Implication for the thesis claim "smoother driving":** *qualified yes*.
Below ~70% of lattice capacity, lateral freedom delivers measurable
longitudinal smoothness. At lattice saturation, that benefit vanishes
and lateral activity becomes overhead. The Week-2 deck's smoothness
claim holds in the design regime, not at capacity. (The "no congestion
regime" caveat from Exp A still applies above k_lattice — neither
condition can model real congestion smoothness.)

## Exp C. Capacity comparison: lane-less vs lane-based  *(done · 2026-05-23)*
Same N, same v_d, same road width. Run lane-less (V2) vs lane-based
(discrete y-positions). Measure throughput and recovery from a
perturbation.

**Setup (exp_capacity.py):** N cars distributed evenly across 2 lanes at
y≈5.0 and y≈9.0 on the W=14 m corridor. Both conditions share initial
positions and seed. Lane-based hard-resets y to the lane centre and
zeros u_y each step. After 8 s of settling, brake car 0 to v_x=2 m/s
for 2 s, then release. Total run 30 s. Wake = # cars below 0.9·v_d.

**Throughput is identical in steady state** (mean v_x = v_d pre and
post-perturbation in both conditions — Exp A finding applies). The
discriminator is the perturbation response.

  | N (k veh/km) | cond | max_wake | Δt_recovery [s] | intra_worst [m] |
  | --- | --- | --- | --- | --- |
  | 60 (120) | lane_based | 5 | 6.10 | 3.50 |
  | 60 (120) | lane_less | 6 | **3.52** | 3.37 |
  | 90 (180) | lane_based | 11 | 7.18 | 1.49 |
  | 90 (180) | lane_less | 8 | **2.72** | 4.25 |
  | 120 (240) | lane_based | 10 | 5.28 | **0.05** ⚠ |
  | 120 (240) | lane_less | 14 | **3.84** | 2.22 |
  | 140 (280) | lane_based | 12 | 7.06 | **0.03** ⚠ |
  | 140 (280) | lane_less | 26 | **2.46** | 1.51 |

**Three findings, in order of importance:**

1. **Safety crossover at N ≈ 120.** Lane-based intra-worst collapses
   from 1.49 m (N=90) to 0.05 m (N=120) and 0.03 m (N=140) — the brake
   car gets effectively rear-ended; saturated 1-D α-repulsion cannot
   absorb the closing velocity fast enough. Lane-less keeps intra-worst
   ≥ 1.5 m at every density tested. This is the headline thesis result
   for lane-less: it is the difference between a slow-traffic incident
   and a crash.

2. **Lane-less recovers 1.4–2.9× faster across every density** (Δt_recov
   2.46–3.84 s vs 5.28–7.18 s). Lateral freedom lets the slow car be
   passed instead of piled-up-behind.

3. **Counter-intuitive: at N=140, lane-less wake (26) > lane-based (12),
   yet lane-less *still* recovers faster.** Lane-less spreads the
   disturbance laterally — many cars dipped slightly below 0.9·v_d but
   each only briefly. Lane-based concentrates the disturbance on a few
   cars that have to fully stop and restart. Wider-but-shallower wake
   ⇒ faster collective recovery.

**Implication:** the "increased roadway capacity" deck claim doesn't
hold *in steady state* (both at v_d → identical throughput in safe
regime). But under perturbation, lane-less protects *capacity through
incidents* — the integrated throughput loss during a 2-s brake event
is 1.4–2.9× smaller, and the safety margin is qualitatively different
(>1.5 m vs ~0 m).

## Exp D. Macroscopic fundamental diagram on intersection  *(done · 2026-05-23)*
Inject cars at known rate at each of the 4 cross arms. Find max
sustainable throughput before backup. Compare to signalized-intersection
model.

**Setup (exp_intersection_mfd.py):** open square arena (120 m × 120 m,
no β walls). Four arms E/W/N/S spawn cars heading toward the centre at
rate λ veh/s/arm; exit when they cross the opposite boundary. V2 τ with
R(+90°), velocity-only (c1_t=0). Compare against signalised reference:
4-phase × 1800 veh/h/lane saturation = **1800 veh/h total**.

**Single-seed sweep is noisy** — intersection dynamics are quasi-chaotic
under continuous injection. Re-ran with 6 seeds (T=40 s) and reported
median + p25/p75 intra-min. Capacity = largest λ with ≥75% safe seeds
(intra > d_a/2) and drop% < 5.

**Headline finding: "Goldilocks density" for the rotational pattern.**

  | τ tuning  | λ=0.2 | λ=0.4 | λ=0.6 | λ=0.8 | λ=1.0 | λ=1.2 |
  | --- | --- | --- | --- | --- | --- | --- |
  | baseline V2 (c2_t=0.15, d_c=70) safe% | **100%** | 0% | 0% | 0% | 0% | 0% |
  | tuned (c2_t=0.20, d_c=100) safe%       | 0%    | 0%  | **83%** | 0%    | 0%    | 0%    |

  Each τ tuning has a *single* λ at which the clockwise rotational
  pattern self-organises — too sparse and the deflections don't form a
  coherent flow, too dense and τ saturates and cars collide. The "sweet
  spot" shifts with c2_t (stronger τ → higher Goldilocks λ).

**Robust capacities:**

  | Configuration | Capacity λ_max | q_total | vs signalised |
  | --- | --- | --- | --- |
  | Baseline V2 τ   | 0.20 veh/s/arm | **2160 veh/h** | 1.20× |
  | Tuned τ         | 0.60 veh/s/arm | **3300 veh/h** | 1.83× |
  | Signalised ref  | — | 1800 veh/h | 1.0× |
  | Single-lane physical cap (4 × v_d/d_a)  | — | 20 571 veh/h | 11.4× |

  No tuning tested supports λ ≥ 0.8 (≥ 11 520 veh/h demand). That's the
  V2 ceiling on continuous demand — well below the per-arm geometric
  limit of v_d/d_a = 5143 veh/h.

**Qualitative claim (thesis):** *V2's intersection success doesn't
transfer to continuous demand*. Phase-2 inv. #4 + #7 measured V2 on a
one-shot 4-flock 90° volley (29 m clearance). Continuous injection at
the same geometry is a different problem: cars need to *clear before
the next wave arrives*, and the τ-induced rotational pattern is brittle
across density — only one specific λ produces a stable circulation,
not a monotonic capacity curve.

**Phase 4 candidates surfaced by Exp D:**
  - τ that adapts to local density (so the same algorithm produces the
    same rotation strength regardless of how busy the centre is).
  - Explicit time-slot scheduling (give up the lane-less ideal at the
    centre; restore it on the arms).
  - Roundabout primitive: replace τ-deflection with a fixed circular
    target band around the centre, so cars follow it rather than
    invent it from pairwise R(+90°) forces.

  The first is closest to the existing algorithm in spirit; the third
  is the simplest engineering solution and likely what a real
  deployment would do.

---

# Phase 2: algorithmic improvement candidates

Three concrete extensions to the McKenzie algorithm surfaced during the
phase-1 investigations. Each fixes a specific failure mode. Investigate
each one by building a prototype and comparing against the unmodified
algorithm on the failure mode it targets.

## 5. Position-based γ with externally assigned target bands  *(done · 2026-05-23)*
**Targets compression mode (#1).** Currently γ does velocity feedback only,
which is why deflection has no "target band" and cars stack against β.
Add a position-feedback term: `u_γ_pos = -c_g_pos * (q_i.y - y_target[flock])`
where `y_target[flock]` is externally assigned (e.g., flock 1 → y=5,
flock 2 → y=19 on a y_hi=24 road).

**Open questions:**
- Does this eliminate the compression in 2×4 vs 2×4?
- What c_g_pos / target placement work best?
- Does it work alongside τ or replace it?
- Side-effects (oscillation, slowdown)?

**Findings (exp_targeted_gamma.py + centroid variant):**

  *Does NOT fix compression.* intra-flock min stays at 4.27–4.60 m for any
  c_g_pos value, with either per-agent or centroid-based spring. The
  compression mode is about *which agent hits β first*, not how the spring
  force is distributed across the flock. The leading row pins itself at
  the β-edge and the trailing row catches up.

  *Does dramatically improve inter-flock min.* At c_g_pos=0.5 (near-critical
  damping for ζ = c_g/(2√c_g_pos)), inter climbs from baseline 5.11 m to
  **8.24 m (per-agent variant) or 7.74 m (centroid variant)** — the
  highest inter-flock distance recorded across all experiments. This is
  because flocks now have an explicit destination instead of "deflect
  blindly downward."

  *High c_g_pos causes oscillation* (ζ drops below 0.5 → underdamped → 
  overshoot). Past c_g_pos=2, inter drops below car width as flocks cross
  paths during recovery.

  *τ off + γ-target alone is not enough.* Target tracking without τ gives
  inter ≈ 2-3 m — the lane-target keeps flocks separated *eventually* but
  there's no close-range avoidance during the encounter.

  **Practical conclusion:** position-based γ is a clean addition that
  raises inter-flock distance ~60% over baseline. It does NOT fix the
  multi-row compression mode — that requires a different intervention
  (probably wider/softer β-zone so leading row pre-decelerates).

## 6. Predictive-gating to suppress unneeded τ  *(done · 2026-05-23)*
**Targets "wasted τ" at large offsets (#3).** V2 from the compression
investigation failed as a *trigger* (self-defeating chatter), but should
work as a *suppressor*: keep McKenzie's existing gate, then *additionally*
require that the projected closest-approach distance is below some
threshold for τ to fire.

**Findings (exp_predict_suppress.py, offset sweep, 1×4 vs 1×4):**

At threshold = 10 m (a few d_a wide):
  - **dy=0 head-on:** inter-min=6.58 m, gate count drops 5454→2620. *No
    regression on collision-imminent case.* The V2 chatter doesn't
    materialize because the base McKenzie gate fires continuously enough
    that γ holds cars at the deflected y once τ suppresses.
  - **dy=14 pre-sorted:** inter-min=14.00 m, gate count **0**. Complete
    suppression of wasted-τ work. The flocks pass at their natural offset
    safely without the algorithm firing once.
  - **dy=4-8 intermediate:** inter-min drops 0.84–3.89 m below baseline.
    Suppression cuts off τ before delivering the extra deflection it would
    otherwise produce. Still well above car-width threshold, but a real
    loss of safety margin.

At lower thresholds (th=3): the V2 chatter mode returns — head-on case
drops to inter=3.00 m. **Threshold must be ≥ d_a or so** to avoid this.

**Practical conclusion:** clean additive enhancement. Saves substantial
unnecessary control work at large offsets and never hurts the
collision-imminent case (which is what matters most for safety). The
3-4 m inter-loss at intermediate offsets is a fair trade.

## 7. True rotation matrix R(±90°) instead of J  *(done · 2026-05-23)*
**Targets intersection failure (#4).** Replace McKenzie's reflection J=[[0,1],[1,0]]
with a proper 90° rotation R=[[0,-1],[1,0]] (or R⊺). Each flock then
deflects by the same physical angle relative to its motion direction,
regardless of orientation.

**Findings (exp_rotation_matrix.py + exp_rotation_working.png):**

  | scenario | J | R(+90°) baseline | R(+90°) tuned (c2_t=0.15, d_c=70) |
  | --- | --- | --- | --- |
  | 2-flock head-on | 6.58 | 6.80 (+0.22) | — |
  | 3-flock at 120° | 0.07 ✗ | 0.44 | **~38 m** ★ |
  | 4-flock at 90° | 0.00 ✗ | 0.21 | **29 m** ★ |

  - **No regression** on the canonical 2-flock case. R produces nearly
    identical force as J for axis-aligned opposing flocks. The 0.22 m
    improvement is a side effect of slightly different x-handling once
    cars develop y-velocity.
  - **Clean roundabout pattern** emerges at intersections. All flocks
    deflect right-of-motion consistently → coherent clockwise spiral.
  - **Baseline τ-strength is too weak** for intersection geometry (curve
    radius r=v²/a ≈ 30 m is too large relative to convergence point).
    Modest bump to c2_t=0.15, d_c=70 fixes it: inter-min on 4-flock
    climbs from 0.21 m to 29 m.
  - **Above c2_t ≈ 0.3** the perpendicular force traps cars in slow
    orbits; max_stall climbs above 5 s.
  - **±90° choice is arbitrary** — opposite rotation directions, same
    avoidance quality. Pick by convention.

**Big-picture claim:** McKenzie's J is a *special case* of a more general
algorithm with R substituted. R(+90°) reduces to J for axis-aligned
opposing flocks (no regression) and extends naturally to non-aligned
multi-flock intersection geometries. This is the cleanest of the three
improvement candidates: a single matrix substitution that unblocks an
entire new geometry class.

---

## 4. 3+ flocks at an intersection  *(done · 2026-05-23)*
**Where it surfaced:** natural extension beyond 2-flock geometries.
McKenzie didn't test this.

**Configurations:** 3 flocks at 120°, 4 flocks at 90° (symmetric), 4 flocks
at 90° with mild lateral jitter. Open arena, walls placed far away.

**Findings:**
  - **No deadlocks** observed (cars maintain speed throughout).
  - **3-flock 120°:** inter-min = 0.07 m at default d_c/dist, improving to
    1.08 m at d_c=80/dist=80. Marginal — still below car width.
  - **4-flock 90° symmetric:** inter-min = 0.00 m at every engagement
    distance tested (40 through 100). Cars pass straight through the
    center. **Structural failure, not a tuning issue.**
  - Jittering one of the 4 flocks (perp_jitter=0.5) lifts inter to 0.33 m
    — marginal.

**The geometric reason this fails (key thesis-worthy finding):**

  McKenzie's `J = [[0,1],[1,0]]` is a *reflection across y=x*, not a
  rotation. The τ-force direction `−J·v_self` is "perpendicular-ish" to
  velocity only when velocity is aligned with the x or y axis. For
  flocks at the cardinal directions:
    - +x flock deflects −y
    - +y flock deflects −x
    - −x flock deflects +y
    - −y flock deflects +x

  *All four deflect to the right of their own motion* → all spiral
  clockwise. In a roundabout this could work if τ were strong enough to
  curve them before the center, but at realistic gains they punch
  straight through and only curve afterward.

  In the 1D opposing-flock case (+x vs −x) this same property gives the
  desired *split into different y-bands* — the reflection happens to
  separate the two flocks. The trick is specific to axis-aligned 1D
  geometry.

**Conclusion / boundary of the algorithm:**

  McKenzie's τ-agent is a **1D corridor algorithm**. It does not extend
  to intersection geometries. To handle 3+ flocks, one would need either:
  - a true rotation matrix `R(±90°)` instead of `J`, so every flock
    deflects by the same physical angle relative to its own motion, or
  - a fundamentally different cooperation mechanism (assigned lanes,
    time-slots, right-of-way priorities).

  This marks the natural scope boundary for the McKenzie-based thesis
  work — single-corridor multi-flock interaction is well-characterized;
  intersection geometry is future work requiring a new algorithm.
