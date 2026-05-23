# Cooperative Flocking for Traffic: Characterization, Algorithmic Improvements, and Traffic-Theory Evaluation

## 1. Introduction

Vehicle flocking, in the sense of Olfati-Saber's α-β-γ framework with McKenzie's lateral-deflection extension τ, proposes that cooperative driving can be coordinated through pairwise forces rather than explicit lane discipline. Each agent receives four contributions: α holds spacing within its own flock, β enforces road boundaries, γ pulls velocity toward a desired direction, and τ produces lateral deflection when two opposing flocks approach. The promise is "lane-less" traffic: smoother trajectories, higher capacity through dense packing, and tolerance to perturbations.

This study characterises the McKenzie algorithm across realistic traffic scenarios and probes the limits of the framework. The work is organised into three phases. Phase 1 stress-tests the published algorithm on geometries that go beyond what McKenzie's paper covered, identifying four failure modes. Phase 2 proposes three concrete algorithmic improvements, each targeting a specific failure mode, and composes them into a mode-aware "V2" algorithm. Phase 3 evaluates the algorithm against established traffic-theory benchmarks: the fundamental diagram of flow versus density, longitudinal smoothness against a lane-locked baseline, capacity through incidents against a lane-based baseline, and the macroscopic fundamental diagram on a four-arm intersection.

The contributions are (1) an empirical characterisation of the McKenzie algorithm's regime boundaries, (2) three orthogonal algorithmic extensions with measured trade-offs, and (3) a four-experiment traffic-theory evaluation that reveals where the lane-less paradigm pays off (incident resilience, smoothness below saturation) and where it does not (steady-state mean throughput, sustained high-demand intersections).

## 2. Methodology

### 2.1 Algorithm

The state of agent i consists of position q_i and velocity p_i in two dimensions. The control input is

  u_i = α + β + γ + τ

with per-step magnitude saturation at a_max = 9 m/s². The four contributions:

* **α** : pairwise gradient and consensus over agents within sigma-norm distance r_a in the same flock, parameterised by lattice spacing d_a, smoothing parameter ε, and gain coefficients (c1_a, c2_a). This produces the lattice equilibrium at separation d_a.
* **β** : a wall-repulsion force from boundaries y = y_lo and y = y_hi, with the same sigma-norm form but parameters d_b, c1_b, c2_b.
* **γ** : velocity feedback to a per-flock desired velocity p_d. In standard form, u_γ = -c_g (p - p_d).
* **τ** : pairwise lateral deflection between opposing flocks within d_c, gated by the heading inner product (only active when relative headings are anti-parallel). McKenzie's original formulation uses J = [[0,1],[1,0]], a reflection across y = x. Phase 2 substitutes R(+90°) = [[0,-1],[1,0]], a true 90-degree rotation.

### 2.2 Common parameters

Across experiments: d_a = 7 m, r_a = 1.2·d_a, c1_a = 5, c2_a = 2√5, d_b = 3 m, c1_b = 200, c2_b = 2√200, c_g = 1.5, ε = 0.1, and v_d = 10 m/s. τ parameters (c1_t, c2_t, d_c) vary by experiment as documented in the individual scripts.

### 2.3 Metrics

Per scenario type, the following metrics recur:

* **Inter-flock minimum**: smallest distance between agents in different flocks across the run. Below car width signals collision.
* **Intra-flock minimum**: smallest pair distance within a flock. Tracks lattice integrity.
* **Max stall**: longest contiguous interval where any agent's speed is below 0.5 v_d. Used to detect deadlocks.
* **RMS acceleration, peak jerk, std(v_x)**: per-car smoothness metrics for Exp B.
* **Wake count and recovery time**: response to a brake perturbation for Exp C.
* **Realised throughput, drop rate, intra-min**: for Exp D's intersection MFD.

### 2.4 Three phases

Phase 1 ran the canonical McKenzie two-flock head-on test, then progressively varied geometry (number of rows, flock-size asymmetry, lateral offset at the start, and three or four flocks meeting at angles). Each variation exposed a specific failure mode that was investigated in isolation.

Phase 2 took each Phase 1 failure mode and built a minimal extension targeting it. Each was tested in isolation against the unmodified algorithm.

Phase 3 dropped the side-by-side ablations and ran four traffic-theory benchmarks. These tests evaluate whether the algorithm's behaviour aligns with traffic-engineering benchmarks such as the Highway Capacity Manual.

## 3. Results

### 3.1 Phase 1: algorithm boundaries

**Compression mode in multi-row flocks.** When two 2×4 flocks meet head-on, the intra-flock minimum drops from d_a = 7 m to ≈ 4.3 m during the encounter. Analysis traced this to the velocity-only τ-force: every agent in a flock shares the same velocity, so all receive identical y-force. The leading row hits β first and gets pinned at the wall; the trailing row, still under the same uniform push, catches up. Four variant τ formulations (hybrid radial, predictive gating, flock-relative direction, projected-radial) all failed to fix this without other regressions. The compression is structural to McKenzie's lateral-deflection scheme inside a hard-walled corridor.

**Asymmetric flock sizes.** With one flock fixed at N₁ = 4 and the other sweeping N₂ ∈ {1, …, 12}, the per-flock deflection magnitude is roughly invariant (≈ 4 m regardless of ratio), because the β-zone edge acts as an emergent target band at d_b = 3 m from each wall. Asymmetry surfaces in wall-proximity time and peak excursion, not collision risk. No N₁/N₂ ratio up to 12 breaks the algorithm in one-dimensional geometries.

**Off-centre initial placement.** With dy sweeps from 0 (head-on) to 14 m (pre-sorted lanes), two counter-intuitive results emerged. First, τ-engagement count *increases* with offset, because at large dy neither flock deflects, so headings stay anti-parallel and the gate stays satisfied longer. Second, deflection magnitude *decreases* with offset because cars start near a wall and reach the β-edge after only ≈ 2 m of motion. Two deflection regimes were identified: encounter-limited (centred, wide road) and β-limited (off-centre or narrow road). This corrects the universal-target-band claim from the asymmetry sweep.

**Three-plus flocks at intersection.** Geometric analysis: McKenzie's J is a reflection across y = x, not a rotation. The τ-force direction is "perpendicular-ish" only when velocity aligns with a cardinal axis. For four flocks at 90°, all four deflect to the right of their motion direction, creating clockwise spiral pressure that is too weak to curve them away from each other at the centre. Symmetric three- and four-flock cases show inter-min of 0.07 m and 0.00 m respectively, well below car width. McKenzie's τ-agent is fundamentally a one-dimensional corridor algorithm; it does not extend to non-cardinal multi-flock geometries without a rotational substitution.

### 3.2 Phase 2: algorithmic improvements

**Position-based γ with externally assigned target bands.** Targets the compression mode. A per-agent position-feedback term u_γ_pos = -c_g_pos (q.y - y_target) was added. The compression itself is not fixed (intra-min stays at 4.27 to 4.60 m for any c_g_pos), but inter-flock minimum climbs dramatically from a baseline 5.11 m to 8.24 m at c_g_pos = 0.5, the highest inter-flock distance recorded across all experiments. The intervention raises inter-flock distance by ≈ 60% without affecting intra-flock compression.

**Predictive-gating suppression.** Targets wasted τ at large offsets. McKenzie's existing gate is kept, with an extra check: τ fires only if the projected closest-approach distance is below a threshold. At threshold 10 m, the head-on dy = 0 case retains inter-min 6.58 m with τ count dropping from 5454 to 2620 (no regression). At dy = 14, τ count drops to 0 (complete suppression of wasted work) with inter-min 14 m. Intermediate offsets lose 0.84 to 3.89 m of inter-min, but all stay safely above car width.

**R(+90°) rotation matrix.** Targets the intersection failure. Substituting R(+90°) for J produces nearly identical force on axis-aligned opposing flocks (J is a special case of R for that geometry), and unlocks the intersection geometry class. Tuned at c2_t = 0.15, d_c = 70: three-flock at 120° climbs from 0.07 m to 38 m; four-flock at 90° climbs from 0.00 m to 29 m. Above c2_t ≈ 0.3, cars get trapped in slow orbits and max_stall climbs above 5 s. This is the cleanest of the three improvements: a single matrix substitution that unlocks a new geometry class with no regression.

**Mode-aware composition (V2).** Combining the three improvements is not trivial. Predictive suppression breaks the R-rotation roundabout pattern, since it turns off τ once projected paths look safe, which is precisely when the curve needs to keep firing. The honest unified algorithm is mode-aware: R(+90°), corrected β, and velocity-only τ are universally on; predictive suppression is corridor-mode only; target γ is opt-in when external lane assignments exist. V2 preserves canonical corridor behaviour, eliminates wall-escapes at high v_d (266 escapes at v_d = 25 reduced to 0), and unlocks the intersection geometry class, at the cost of 1 to 4 m of inter-flock margin sacrificed in cases that were already safely above car width.

### 3.3 Phase 3: traffic-theory benchmarks

**Exp A. Fundamental diagram.** A periodic corridor (L = 500 m, W = 14 m) sweep of N ∈ [10, 700] revealed that the classical q-k-v fundamental diagram framing is degenerate for this algorithm. In a translation-invariant periodic corridor the α-gradient is x-symmetric, so its mean x-component is zero at steady state. γ is the only x-asymmetric force, so mean(γ_x) = 0 implies mean(v_x) = v_d at every density. This was confirmed empirically: mean forward speed equals 36.0000 km/h at k = 80, 240, 500, 800, and 1400 veh/km. Past the lattice capacity the algorithm fails by overlap (intra-min drops to zero), not by slowing.

Reframed as a max-stable-density experiment, two thresholds were extracted:

| Metric | Threshold | k [veh/km] | q = k·v_d [veh/h] |
| --- | --- | --- | --- |
| k_lattice (lattice intact) | intra-min ≥ 0.9·d_a | 280 (N = 140) | 10 080 |
| k_stable (no overlap) | intra-min ≥ d_a/2 | 320 (N = 160) | 11 520 |
| Strip-hex theory | 2 rows × L/d_a | 286 | (geometric) |
| HCM equivalent | 2200/lane × (14 / 3.6) | (n/a) | 8 556 |

k_lattice matches strip-hex theory within 2%. The algorithm achieves 1.18× HCM at the lattice-intact threshold and 1.35× HCM at the no-overlap threshold, but the comparison is qualified by the lack of a congestion regime.

**Exp B. Smoothness against lane-locked baseline.** Same corridor, two conditions with shared initial state: flocking (full α+β+γ) versus lane-locked (y-velocity zeroed each step). Two regimes emerged.

| N (k veh/km) | rms_ax flock | rms_ax lock | std_vx flock | std_vx lock | peak_jx flock | peak_jx lock |
| --- | --- | --- | --- | --- | --- | --- |
| 90 (180) | 0.064 | **0.183** | 0.018 | **0.196** | 3.4 | 5.4 |
| 120 (240) | 0.017 | **0.111** | 0.005 | **0.129** | 1.1 | 1.6 |
| 140 (280) | 0.095 | 0.105 | 0.079 | 0.086 | **3.7** | 1.6 |
| 160 (320) | 0.119 | 0.124 | 0.137 | 0.143 | **3.6** | 1.4 |

Below the lattice-saturation density (N ≤ 120), flocking has 3 to 26 times lower rms_a_x and std(v_x) than lane-locked. Cars resolve spacing imbalances by shifting laterally rather than braking. At and above the lattice limit (N ≥ 140) the advantage vanishes; rms_a_x and std(v_x) converge, and flocking peak |jerk_x| becomes 2 to 3 times *worse* than lane-locked, because lateral coupling injects sharper x-jerks once there is no room to manoeuvre. The crossover at N = 140 (k = 280 veh/km) matches Exp A's k_lattice within rounding error.

**Exp C. Lane-less versus lane-based perturbation recovery.** Both conditions use two lanes (y ≈ 5 and y ≈ 9 in a W = 14 m corridor) and share initial positions. After 8 s of settling, car zero is held at v_x = 2 m/s for 2 s, then released.

| N (k veh/km) | condition | max_wake | Δt_recovery [s] | intra_worst [m] |
| --- | --- | --- | --- | --- |
| 60 (120) | lane_based | 5 | 6.10 | 3.50 |
| 60 (120) | lane_less | 6 | **3.52** | 3.37 |
| 90 (180) | lane_based | 11 | 7.18 | 1.49 |
| 90 (180) | lane_less | 8 | **2.72** | 4.25 |
| 120 (240) | lane_based | 10 | 5.28 | **0.05 ⚠** |
| 120 (240) | lane_less | 14 | **3.84** | 2.22 |
| 140 (280) | lane_based | 12 | 7.06 | **0.03 ⚠** |
| 140 (280) | lane_less | 26 | **2.46** | 1.51 |

Three findings. First, lane-based intra-worst collapses to 0.05 m at N = 120 and 0.03 m at N = 140; the brake car gets effectively rear-ended because saturated one-dimensional α-repulsion cannot absorb the closing velocity. Lane-less stays ≥ 1.5 m at every density tested. Second, lane-less recovers 1.4 to 2.9 times faster across all densities. Third, at N = 140 lane-less has a *larger* wake (26 cars) than lane-based (12 cars) yet still recovers faster: the disturbance spreads laterally across many cars briefly affected, instead of concentrating on a few that must fully stop.

**Exp D. Intersection macroscopic fundamental diagram.** Open arena (120 × 120 m), four arms inject cars at rate λ veh/s/arm toward the centre. Six-seed averages with safety threshold intra > d_a/2:

| τ tuning | λ = 0.2 | λ = 0.4 | λ = 0.6 | λ = 0.8 | λ = 1.0 | λ = 1.2 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (c2_t = 0.15, d_c = 70) safe% | **100%** | 0% | 0% | 0% | 0% | 0% |
| tuned (c2_t = 0.20, d_c = 100) safe% | 0% | 0% | **83%** | 0% | 0% | 0% |

Each τ tuning has exactly one λ at which the rotational pattern self-organises. Baseline V2 is robust only at λ = 0.2 (2 160 veh/h total, 1.20× signalised). Tuned V2 is robust only at λ = 0.6 (3 300 veh/h, 1.83× signalised). No tuning tested supports λ ≥ 0.8. All values are well below the per-arm geometric maximum (v_d/d_a × 4 = 20 571 veh/h). The non-monotonic "Goldilocks" behaviour contradicts the expectation of a monotonic capacity curve.

## 4. Discussion

### 4.1 Cross-experiment observations

**Two independent experiments converge on k = 280 veh/km.** Exp A measured the largest density at which the α-lattice survives in a steady-state corridor (k_lattice = 280). Exp B measured the largest density at which lateral freedom delivers a smoothness benefit (crossover at the same N = 140, k = 280). The matching boundary is structural: it is the strip-hex packing limit. Below it cars have spare room and the algorithm has degrees of freedom to exploit; at and above it the lattice is full and lateral manoeuvring becomes counter-productive.

**McKenzie's algorithm is not a traffic-flow model.** Exp A reframes it: the algorithm is a steady-state cruise controller for a fleet of agents already at v_d. In a translation-invariant setting the constant-v_d γ force pins mean speed to v_d regardless of density. There is no mechanism for collective slowdown when blocked. This is reflected in the lack of a fundamental-diagram inflection, and in the lattice failing by overlap rather than by speed reduction.

**Lane-less protects through incidents, not in steady state.** Exp C confirms that steady-state throughput is identical between lane-less and lane-based conditions in the safe regime (both at v_d), but lane-less prevents rear-end near-collisions at high density and recovers 1.4 to 2.9 times faster from a brake perturbation. The often-stated "increased roadway capacity" claim does not hold in steady state, but lane-less protects capacity through incidents in a way that lane-based control cannot.

**V2 succeeds at one-shot intersections but fails at continuous demand.** Phase 2 measured V2 on a single-volley four-flock 90° encounter (29 m inter-min). Exp D under continuous injection at the same geometry shows the algorithm cannot sustain even moderate demand: each tuning has a single Goldilocks density. The pairwise R(+90°) deflection that worked for one volley does not robustly self-organise a continuous flow.

### 4.2 Phase 4 candidates

The Phase 3 experiments expose three algorithmic gaps that future work could address.

* A **car-following extension** would let γ's target shrink when the agent is blocked, allowing collective slowdown at high density. A single algorithmic addition that would let the algorithm model congestion and let Exp B and Exp D measure smoothness and throughput in stop-and-go conditions, not just at v_d.
* **Density-adaptive τ** would let deflection strength scale with local agent density, producing the same rotation strength regardless of how busy the centre is. This addresses Exp D's Goldilocks behaviour.
* A **roundabout primitive** would replace pairwise τ-deflection at intersections with a fixed circular target band that cars follow. This is the simplest engineering solution for sustained intersection demand and is closest to what a real deployment would do.

### 4.3 Scope of the algorithm and the lane-less paradigm

The work delineates a working envelope for the McKenzie algorithm: it characterises one-dimensional corridor flows from low density up to the strip-hex packing limit, with measurable smoothness and incident-resilience advantages over lane-based control inside that regime. It does not characterise congested flow, and its intersection performance is limited to one-shot encounters or a single Goldilocks density under continuous demand.

The Phase 2 V2 algorithm extends this envelope but does not change its boundary fundamentally. R(+90°) is a clean win that unlocks the intersection geometry class for one-shot tests. Predictive suppression and target γ are clean additive improvements within the corridor regime. None of the three addresses the lack of a slowdown mechanism, which is the root constraint behind all four Phase 3 experiments.

The lane-less paradigm pays off where Phase 3 measured advantages: incident resilience (Exp C) and longitudinal smoothness below saturation (Exp B). It does not pay off where Phase 3 found no advantage or active disadvantage: steady-state mean throughput (Exp A, identical to v_d·k), high-density smoothness (Exp B, peak-jerk regressions), and sustained intersection capacity (Exp D, Goldilocks fragility). These results suggest the lane-less paradigm is best deployed as an incident-resilience and smoothness layer on top of conventional traffic engineering, not as a replacement for it.
