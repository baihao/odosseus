# Parameters of the final method: what is tuned, what can be derived, what can be estimated online

Every number in `odo/online.py: DEFAULT` is an engineering decision that needs a justification. This document sorts
all of them by the kind of justification they can have. It proposes how to replace tuned values with values that are
**measured on the training run**, **derived from the filter's own covariance**, or **estimated online**. It also
gives the evidence behind each proposal. The proposals are not implemented yet; section 4 gives the order and how to
check each step.

Evidence files: [`outputs/24_day/filter_consistency.md`](../outputs/24_day/filter_consistency.md)
(`scripts/filter_consistency.py`) and [`outputs/24_day/dr_calibration.md`](../outputs/24_day/dr_calibration.md)
(`scripts/dr_calibration.py`).

## 1. Summary

`DEFAULT` has 44 entries. Sorted by justification:

| class | count | what it means | parameters |
|---|---|---|---|
| **A. switches for experiments** | 9 | off in the final system; remove from `DEFAULT` and keep in the test harness | `r_inl_ref`, `r_dist_ref`, `sim1_lo`, `lo_below`, `mask_bright`, `kp_keep`, `chal_fixed`, `reloc`, `log_sim` (and per-run inputs `img_root`, `aug`, `kidnap`, `init_offset`, `init_pose`) |
| **B. statistical constant** | 1 | a χ² quantile, not a tuning value | `gate` |
| **C. measured on the training run** (calibration, uses training GT, allowed) | 9 | one script run, no lap-error optimisation | `q_lon`, `q_lat`, `q0`, `q_yaw`, `r0_pos`, `r0_yaw`, `prior_rp_deg`, `prior_z`, `max_gt_acc` (a rule on GT metadata) |
| **D. derived from the filter covariance or another parameter** | 12 | disappear as separate values | `r1`, `r1_grow`, `reloc_r1`, `r2`, `sel_radius`, `reloc_after`, `reloc_sig0`, `reloc_sig_rate`, `reloc_yaw0_deg`, `reloc_yaw_rate_deg`, `chal_m`, `chal_yaw` |
| **E. estimated online** | 2 | from the current frame's own residuals | `huber`, `r_scale` (see 3.3: the online estimate must not be used alone) |
| **F. compute budget** | 4 | runtime / coverage trade-off, set by the 100 ms frame budget | `max_kp`, `n_pass`, `reloc_every`, `extra_dof` (model choice) |
| **G. genuine choices left** | 7 → 3 | can be reduced to one false-match level, one descriptor threshold and one persistence count | `sim1`, `sim2`, `min_inl`, `yaw_min_inl`, `reloc_min_inl`, `reloc_strong_inl`, `chal_n` |

So 37 of the 44 entries need no tuning once the proposals are in place. The core is about three values: descriptor similarity, a false-match level and
how many consistent fixes it takes to override the filter. All three are about matching, which is the one part with
no physical model. The rest can be justified with measurements or statistics.

This fits what tuning showed so far. The stage-4 sweep of matching thresholds, map passes, noise floor and
keypoints was flat (0.103–0.108 m, [`experiments_explained.md`](experiments_explained.md) section 6.4). Every change that
mattered (the recovery rules in Exp21 and Exp24) fixed a failure mode where the filter trusted a wrong measurement. Those
rules are exactly the class-D parameters: hand-made stand-ins for a covariance that the filter does not model correctly.

## 2. Evidence

### 2.1 Dead-reckoning drift (car state vs training GT)

The car state (vx, vy, wz) integrated from GT poses at 3000 random times on the night laps L1–L8. This uses the same
integration code as the filter.

| horizon | distance | lon σ | lat σ | position p95 | yaw σ | position p95 / s | yaw σ / s |
|---|---|---|---|---|---|---|---|
| 0.1 s | 2.2 m | 0.012 m | 0.011 m | 0.034 m | 0.012° | 0.34 m/s | 0.12°/s |
| 1 s | 22 m | 0.088 m | 0.099 m | 0.295 m | 0.070° | 0.30 m/s | 0.070°/s |
| 2 s | 44 m | 0.165 m | 0.220 m | 0.588 m | 0.132° | 0.29 m/s | 0.066°/s |
| 5 s | 107 m | 0.416 m | 0.581 m | 1.495 m | 0.255° | 0.30 m/s | 0.051°/s |
| 10 s | 204 m | 0.855 m | 0.966 m | 2.564 m | 0.289° | 0.26 m/s | 0.029°/s |

* **Position drift grows linearly with time**, about 0.3 m/s at p95 (0.4–0.8 % of the distance driven). That is the signature
  of a slowly varying bias (speed scale, yaw-rate bias), not of white noise.
* **The filter's process noise `Q` models a random walk** (variance added per 0.1 s step, so σ grows with √t). It
  matches the measurement up to about 1 s (lon 0.086 vs 0.088 m; lat 0.051 vs 0.099 m) but is too small by 2× at 2 s and
  3–6× at 5–10 s:

  | horizon | Q lon / lat σ | measured lon / lat σ |
  |---|---|---|
  | 1 s | 0.086 / 0.051 m | 0.088 / 0.099 m |
  | 5 s | 0.187 / 0.111 m | 0.416 / 0.581 m |
  | 10 s | 0.254 / 0.152 m | 0.855 / 0.966 m |

  So after a few seconds without a fix, the filter is overconfident. That is why the relocalisation plausibility rules
  (`reloc_sig*`, `reloc_yaw*`) had to be added on top of the gate.
* **Correction to [`final_method.md`](final_method.md):** it states the position rule `0.5 m + 0.3 m/s` keeps a 5× margin over
  a drift of ~0.06 m/s. The measured p95 drift rate is ~0.3 m/s, so 0.3 m/s is about the p95 itself, with no margin; the
  0.5 m offset adds some at short gaps. (0.06 m/s came from the drift averaged over two whole laps, where errors
  partly cancel around the loop.) The yaw rule `0.3° + 0.1°/s` keeps ~3–4× over the measured σ at 1–10 s.

### 2.2 Is the filter consistent? (NIS and NEES)

NIS = normalised innovation of each fix that reaches the gate. With a correct 3-DoF filter it has mean 3 and exceeds the gate 16 in
~0.1 % of fixes. NEES = position error vs GT, normalised by the filter's σ (x, y). A correct filter gives mean ~2 and
values above 9.21 in ~1 %.

| run | pos RMSE | NIS mean / median | NIS > 16 | NEES mean / median | NEES > 9.21 |
|---|---|---|---|---|---|
| F1/L5 night, 7 cams (tuning lap) | 0.089 m | 0.6 / 0.3 | 0.0 % | 4.5 / 2.2 | 7.8 % |
| F7/L2 night (lap in the map) | 0.067 m | 0.5 / 0.2 | 0.0 % | 4.5 / 0.7 | 8.9 % |
| F5/L2 night, 7 cams | 0.299 m | 2.1 / 0.6 | 2.0 % | 45.9 / 7.3 | 44.5 % |
| F5/L2 night, cl + cr | 0.287 m | 4.4 / 1.0 | 5.8 % | 36.0 / 11.5 | 54.0 % |
| F5/L2 day (CycleGAN, lamps removed), cl + cr | 0.265 m | 5.2 / 0.9 | 7.6 % | 28.1 / 9.7 | 50.8 % |
| F5/L2 day_qwen, cl + cr | 0.419 m | 6.2 / 2.4 | 8.6 % | 17.2 / 7.0 | 44.4 % |

* **Innovations are too small** (NIS median 0.2–0.6 on well-mapped laps): the measurement covariance, inflated by
  `r_scale` 4 and the 0.10 m floor, is larger than the fixes' scatter around the prediction.
* **The errors are too large** (NEES mean 4.5 even on the best laps, up to 46 on L2): the true error is 1.5–5× the σ
  the filter believes. On the off-line lap L2, the filter is badly overconfident.
* Both together mean **the fixes are biased, not noisy**. Consecutive fixes agree with each other and with dead
  reckoning, but are all off the GT in the same direction for seconds. That is the map/GT error floor (~0.1 m, repeating at the same
  corners on every lap), and on L2 the extrapolation 8–14 m away from the mapped lines.
* Consequence: a bias cannot be seen from innovations. Any *online* noise estimation that matches innovations
  (covariance matching, Sage–Husa) would *shrink* R and make the filter even more overconfident. The floor `r0_pos`
  must come from training data where GT is available (section 3.1). Only then can the rest be derived from P.

## 3. Proposals, parameter by parameter

### 3.1 Class C: measure once on the training run

| parameter | now | measure from | how |
|---|---|---|---|
| `q_lon`, `q_lat`, `q0` | 0.01, 0.005 per m, 0.005 m | 2.1 | replace the random-walk Q by a model that fits the measured σ(h): add a speed-scale bias and a yaw-rate bias as slow states (Gauss–Markov, time constant from the drift curve), or fit Q's per-metre terms to the 0.1–1 s rows, where most frames are (fixes arrive every 0.1 s on 91–100 % of frames) |
| `q_yaw` | 0.0005 rad per step | 2.1 | yaw σ 0.012° per 0.1 s, 0.07° per 1 s, with the slow bias state above |
| `r0_pos`, `r0_yaw` | 0.10 m, 0.23° | map consistency | leave-one-lap-out fixes on the map laps vs GT (already run as CV): σ of the fix error when the filter is converged. The error floor measured there is 0.07–0.12 m per lap. This is the bias that innovations cannot see (2.2) |
| `prior_rp_deg`, `prior_z` | 0.5°, 0.05 m | training GT attitude | distribution of roll / pitch / height of the car relative to the nearest map frame on the same spot of another lap (state.csv has roll, pitch, z) |
| `max_gt_acc` | 1.0 m | GT metadata | not a tuning value: map frames whose own GT accuracy flag is above the tolerance of the method (1 m ≫ the 0.1 m floor) are dropped |

### 3.2 Class D: derive from the filter's covariance (P) and the PnP residual

Once Q and the floor are calibrated, these lose their own value:

| parameter | now | derived as |
|---|---|---|
| `r1`, `r1_grow`, `reloc_r1` | 24 px, +40 px per s without fix, 60 px | per map point, the 3σ ellipse of its projection under P: `r = 3·sqrt(J P Jᵀ + σ_px²)`. Near points get large windows, far points small ones. `r1` 24 px ≈ 3σ today for points ≥ ~7 m away (σ_pos 0.06 m, f ≈ 1000 px). The time growth `r1_grow` is a hand-made stand-in for P growing without fixes |
| `r2` | 4 px | 3 × the stage-1 reprojection rms (median 1.3 px on night and day runs) |
| `sel_radius` | 15 m | 3σ of the predicted position plus the lateral spread of the map passes at that spot (lines up to 14 m apart on L2). The 15 m that fixed L2 is about that spread |
| `reloc_after` | 1 s | relocalise when the 3σ projection window exceeds what guided matching can search (P-based), not after a fixed time |
| `reloc_sig0`, `reloc_sig_rate`, `reloc_yaw0_deg`, `reloc_yaw_rate_deg` | 0.5 m + 0.3 m/s; 0.3° + 0.1°/s | with a calibrated Q (drift ~0.3 m/s, 0.03–0.07°/s yaw), the filter's own P after a gap already has the right size, and the ordinary χ² gate decides on a relocalisation jump. The four values vanish |
| `chal_m`, `chal_yaw` | 0.3 m, 0.29° | agreement of two fixes = χ² test with their two measurement covariances plus DR drift between them |

### 3.3 Class E: estimate online, from the current frame

| parameter | now | online estimate |
|---|---|---|
| `huber` | 2 px | robust scale of the PnP residuals of this frame (1.4826 × median absolute residual, ≈ 1.3 px), Huber at 1.5 × that scale. This adapts to blur and day images without a fixed value |
| `r_scale` | 4 | the PnP covariance (inverse Hessian) is optimistic because inliers are not independent. Its online part (inflation from the residual scale above, times the effective number of independent inliers) can be estimated per frame. The bias floor `r0_pos` stays the calibrated value from 3.1, never an online estimate (2.2) |

### 3.4 Class F: compute budget, not accuracy

`max_kp` (2048), `n_pass` (3), `reloc_every` (0.5 s): set by the 100 ms frame budget (now 33–41 ms with 7 cameras).
The flat sweep showed accuracy does not depend on them in the tested range. `extra_dof` (6-DoF PnP with priors) is a
model choice backed by an experiment (lateral −10 %), not a number.

### 3.5 Class G: the genuine choices, reduced to three

* **One false-match level** instead of four inlier counts (`min_inl` 30, `yaw_min_inl` 60, `reloc_min_inl` 80,
  `reloc_strong_inl` 300). Each asks the same question: could this many inliers have come from wrong matches? An
  a-contrario test answers it: the expected number of random matches that land within the inlier radius (3 px)
  given the number of candidate matches, the window area and the number of hypotheses tried (1 for tracking, 5 for
  relocalisation). A fix is accepted when this number is below one value ε, and the threshold then grows with the
  size of the search by itself. The yaw rule becomes "use yaw when its PnP variance is below the
  car-state yaw drift since the last fix", which is P-based (class D).
* **One descriptor threshold.** `sim2` (0.6) is used only within 4 px of an already solved pose; it can be fixed
  relative to `sim1` or dropped (mutual best match only). `sim1` (0.75) is the real one: it decides how many day
  matches survive (Qwen: median 28 matches vs 532 at night). An online alternative (a per-frame threshold from the
  similarity distribution of in-window vs out-of-window pairs) is possible but untested, and the related
  `sim1_lo` hurt night accuracy (0.29 → 0.68 m).
* **One persistence count** (`chal_n` 8): how many consistent rejected fixes override the filter. As a sequential
  probability ratio test between the filter and the challenger, it follows from the same ε, so it is not separate either.

## 4. Order of work and how to check it

1. **Calibrate Q** with the bias states (3.1), using `scripts/dr_calibration.py` as the test: the filter's predicted σ(h)
   must match the measured table at 0.1–10 s. Then **drop the four relocalisation plausibility values** (3.2) and use
   the gate. Check: the Exp24 failures must still be caught (Qwen F5/L2 21 m jump, keep 10 %, gain 1.6), kidnaps
   3–80 m must recover, and the error on F5/L2 night/day/Qwen, F1/L5 and F5/L7 must not change.
2. **Floor from map consistency** (3.1), then check NEES on the same runs. Target: NEES mean near 2 on the mapped laps
   (now 4.5).
3. **P-based windows and selection** (3.2), replacing `r1`, `r1_grow`, `reloc_r1`, `r2`, `sel_radius`, `reloc_after`.
4. **Online residual scale** for `huber` and the frame's covariance (3.3).
5. **A-contrario acceptance** (3.5), replacing the four inlier counts and `chal_n`.

Each step removes parameters, so each step is checked on its own against the frozen results above. Per the protocol
there is no full CV: F5/L2 (night, day, Qwen), F1/L5 and one untouched lap (F5/L7). A step is kept if it removes
parameters without making any of these worse beyond noise. It need not improve them.
