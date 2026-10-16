# Phase 1 Simulation Spec: Shared-Autonomy Person Following

## 1. Purpose

Phase 1 delivers a simulation-based design and analysis of the two ground-station control loops (yaw and range) for the BetaFPV Meteor65 Pro person-following system. The hardware build moves to a later phase.

The simulation answers three questions:

1. Do the yaw and range controllers track a walking person with acceptable error?
2. How much total delay (vision plus command link) can each loop tolerate before it loses stability margin?
3. How does keypoint noise, which grows with range, limit range tracking?

## 2. System Being Modeled

The system uses shared autonomy. The pilot controls altitude and roll. The ground station controls yaw and range. Betaflight runs attitude stabilization onboard, so the ground station commands a drone that is already stabilized:

- **Yaw:** Betaflight always controls yaw as a rate. The ground station sends a yaw-rate command.
- **Pitch:** Betaflight runs in angle mode. The ground station sends a pitch command (a target pitch angle).
- **Altitude and roll:** The pilot holds these. The simulation assumes the pilot holds them exactly: constant altitude, level roll, zero sideways drift.

Out of scope for Phase 1:

- Motor dynamics and full six-degree-of-freedom (6-DOF) rigid-body dynamics
- Thrust loss from tilting (about 3.4% at 15 degrees; the pilot absorbs it through altitude control)
- Battery voltage sag
- Rendered camera images, lens distortion, and camera tilt as the drone pitches
- Mapping physical commands to Betaflight stick values (the command link is not chosen yet)

## 3. Models

### 3.1 Coordinates and Sign Conventions

- World frame: planar, x east, y north, meters.
- Heading ψ: radians, measured counterclockwise from +x.
- Bearing β: angle of the person relative to the drone's nose, positive when the person is to the left.
- Pitch angle θ: positive when the nose tilts down, which accelerates the drone forward.
- Range r: straight-line distance from drone to person, meters.

Tests in Section 10 enforce these conventions.

### 3.2 Plant (Nonlinear, Planar)

States: x, y, ψ, ω (yaw rate), v (forward speed), θ.

```
dω/dt = (ω_c − ω) / τ_ψ
dψ/dt = ω
dθ/dt = (θ_c − θ) / τ_θ
dv/dt = g·tan(θ) − k_v·v + a_d(t)
dx/dt = v·cos(ψ)
dy/dt = v·sin(ψ)
```

- ω_c is the yaw-rate command and θ_c is the pitch command, both after the command link (Section 3.5).
- τ_ψ and τ_θ are first-order time constants that approximate Betaflight's onboard yaw-rate and angle loops.
- k_v is linear drag per unit mass (1/s), so the model needs no vehicle mass.
- a_d(t) is an optional forward disturbance acceleration (gust), zero by default.
- Saturate θ_c to ±θ_max and ω_c to ±ω_max at the plant input.
- Integrate with fixed-step RK4 at dt_phys = 1 ms.

### 3.3 Linear Models (for Frequency-Domain Analysis)

Yaw loop, from yaw-rate command to heading:

```
G_ψ(s) = 1 / ( s·(τ_ψ·s + 1) )
```

Range loop, from pitch command to forward position:

```
G_r(s) = g / ( s·(s + k_v)·(τ_θ·s + 1) )
```

Range decreases as forward position increases, so the range controller's positive sign closes a negative feedback loop.

Open-loop transfer function: L(s) = C(s)·G(s)·e^(−s·T_eff), where the effective delay is

```
T_eff = T_vision + T_link + 0.5 / f_cam
```

The last term approximates the delay a zero-order hold adds at the camera frame rate. Apply the delay exactly in the frequency response (multiply by e^(−jωT_eff)) rather than with a Padé approximation. Compute gain margin and phase margin from the frequency response arrays.

### 3.4 Sensor Model

Pinhole camera with horizontal field of view HFOV and image width W pixels. Focal length in pixels:

```
f = (W / 2) / tan(HFOV / 2)
```

Torso length L is the distance from the shoulder midpoint to the hip midpoint (meters).

Range measurement:

```
d   = f·L / r_true + n_d,      n_d ~ N(0, σ_d²)
r_m = f·L / d
```

Bearing measurement (u is the torso center's horizontal pixel, measured from the left edge):

```
u   = W/2 − f·tan(β_true) + n_u,   n_u ~ N(0, σ_u²)
β_m = atan( (W/2 − u) / f )
```

Detection rules:

- No measurement when |β_true| > HFOV/2 (person outside the frame).
- No measurement when d < d_min (person too far for reliable keypoints).
- Optional random dropout with probability p_drop per frame (default 0).

Timing:

- Generate measurements at frame rate f_cam.
- Each measurement reaches the controller T_vision seconds after capture. Implement this with a timestamped queue.

### 3.5 Command Link

- The controller output passes through delay T_link and updates at f_link with zero-order hold.
- Commands stay in physical units (rad/s for yaw rate, rad for pitch).
- Stick mapping waits until the team chooses the command link.

### 3.6 Person Motion

Each scenario function returns person position and velocity as functions of time (Section 6).

## 4. Estimator and Controllers

### 4.1 Range Estimator

Constant-velocity Kalman filter with state [r, dr/dt]:

- Measurement: r_m, with variance R = (r̂²·σ_d / (f·L))². This propagates pixel noise through the inverse range mapping, so measurement noise grows with the square of range.
- Process noise: white acceleration with spectral density q_a (tunable).
- Update on each new measurement. The prediction step uses the time since the last measurement.
- During dropouts, predict without updating.
- Option `predict_latency` (default off): predict the estimate forward by T_vision before handing it to the controller.

Baseline for comparison: range rate from a finite difference of raw r_m.

### 4.2 Yaw Controller

```
ω_c = K_p,ψ·β_m + K_i,ψ·∫β_m dt
```

- Baseline: P-I, as in the project design.
- Comparison: P-only. The yaw plant contains an integrator, so P-only already drives a stationary bearing error to zero. The integral term removes the steady error when the person walks across the frame (a ramp in bearing). The comparison shows whether that improvement justifies the phase the integral term costs.
- Clamp the integrator (anti-windup). Saturate the output to ±ω_max.

### 4.3 Range Controller

```
e   = r̂ − r_ref          (positive when the drone is too far)
θ_c = K_p,r·e + K_d,r·(dr̂/dt) + K_i,r·∫e dt
```

- K_i,r defaults to 0.
- The derivative term uses the estimator's range rate, not a raw difference.
- Clamp the integrator. Saturate the output to ±θ_max.

### 4.4 Dropout Behavior

When no measurement arrives for longer than `loss_timeout`, both controllers command zero and reset their integrators. Before the timeout, the estimator coasts and the controllers keep running on its output (yaw holds its last command).

### 4.5 Interface

`controllers.py` and `estimator.py` import only numpy and the standard library. Each class exposes `update(measurement, t) -> command` and `reset()`. The ground station will import these files directly, so the gains and filter tuned here are the code that flies.

### 4.6 Gain Design

`scripts/tune_gains.py`:

1. Use nominal parameters and the nominal effective delay.
2. Grid-search gains on the linear model for phase margin ≥ 45 degrees and gain margin ≥ 6 dB, maximizing crossover frequency.
3. Write the chosen gains to `config/gains.yaml` and print margins and crossover frequency.

Then run the nonlinear simulation with noise. If noise makes the pitch command chatter (high RMS pitch command at steady state), lower K_d,r or raise estimator smoothing, and record the trade in the report.

## 5. Parameters (`config/params.yaml`)

Every value below except g is an assumption. Replace values with vendor data or measurements as they become available, and keep the sweep ranges for sensitivity analysis.

| Parameter | Symbol | Nominal | Sweep | Basis |
|---|---|---|---|---|
| Gravity | g | 9.81 m/s² | fixed | physical constant |
| Yaw-rate time constant | τ_ψ | 0.05 s | 0.02 to 0.15 s | assumed |
| Angle-loop time constant | τ_θ | 0.05 s | 0.02 to 0.15 s | assumed |
| Linear drag per unit mass | k_v | 0.5 1/s | 0.2 to 1.5 1/s | assumed |
| Max pitch command | θ_max | 15 deg | fixed | placeholder; use proposal safety plan value |
| Max yaw-rate command | ω_max | 90 deg/s | fixed | placeholder; use proposal safety plan value |
| Camera horizontal field of view | HFOV | 120 deg | fixed | assumed; check camera spec |
| Image width | W | 640 px | fixed | capture resolution |
| Torso length | L | 0.50 m | 0.40 to 0.60 m | assumed adult |
| Separation pixel noise | σ_d | 1.5 px | 0.5 to 4 px | assumed until webcam test |
| Horizontal pixel noise | σ_u | 2.0 px | 0.5 to 5 px | assumed until webcam test |
| Minimum detectable separation | d_min | 10 px | fixed | assumed |
| Camera frame rate | f_cam | 30 Hz | 15 to 30 Hz | analog video |
| Vision latency | T_vision | 0.100 s | 0.050 to 0.300 s | assumed |
| Command link update rate | f_link | 50 Hz | fixed | assumed |
| Command link latency | T_link | 0.020 s | 0 to 0.050 s | assumed |
| Range setpoint | r_ref | 3.0 m | fixed | placeholder; use proposal value |
| Loss timeout | loss_timeout | 0.5 s | fixed | assumed |
| Random dropout probability | p_drop | 0 | 0 to 0.1 | assumed |

At nominal values, f ≈ 185 px. The torso spans about 31 px at 3 m, and range noise is about 0.15 m at 3 m and 0.41 m at 5 m.

Config keys holding angles end in `_deg`. Code converts to radians on load.

## 6. Scenarios

All scenarios run 20 s. The drone starts at the origin facing +x.

| ID | Name | Setup |
|---|---|---|
| S1 | Yaw step | Person stationary at r_ref; drone heading starts 20 deg off. |
| S2 | Range step | Person stationary, aligned; drone starts at r_ref + 1 m. Second run at r_ref − 1 m. |
| S3 | Walk away | Person starts at r_ref ahead, walks directly away at 1.4 m/s for 8 s, then stops. |
| S4 | Walk across | Person starts at r_ref ahead, walks perpendicular at 1.0 m/s for 8 s, then stops. |
| S5 | Walk toward | Person starts at r_ref + 2 m ahead, walks toward the drone at 1.0 m/s for 4 s, then stops. |

Seed all noise. For metrics, run 20 seeds per scenario and report mean and standard deviation.

## 7. Metrics (`results/metrics.csv`)

Per run:

- Overshoot (%) for S1 and S2
- Settling time into a band of ±0.2 m (range) or ±3 deg (bearing)
- RMS and maximum tracking error after t = 2 s
- RMS and peak command; fraction of time saturated
- Target loss: whether the loss timeout triggered

## 8. Figures

Priority order. Finish all Must figures before starting Should figures.

**Must**

- **F1 Sensor model.** (a) Torso pixel separation vs. range with ±1σ band. (b) Range standard deviation vs. range, analytic curve plus Monte Carlo points.
- **F2 Open-loop Bode plots** for yaw and range at nominal delay, with gain and phase margins annotated.
- **F3 Step responses** for S1 and S2: nonlinear simulation with and without noise, overlaid on the linear model.
- **F4 Walk away (S3).** True, measured, and estimated range with setpoint; pitch command; drone and person speed.
- **F5 Walk across (S4).** True and measured bearing; yaw-rate command; top-down trajectory of drone and person with heading arrows.
- **F6 Delay sweep.** Phase margin vs. total delay for both loops, and RMS error vs. total delay from simulation. Mark the maximum tolerable delay, defined as the delay where phase margin falls to 30 degrees.

**Should**

- **F7 Estimator comparison** on S3: Kalman filter range rate vs. finite difference, and the resulting pitch command chatter.
- **F8 Sensitivity.** RMS error vs. τ_θ, k_v, and σ_d, as small multiples.

**Could**

- **F9 Yaw P vs. P-I** on S4.
- **F10 Gust rejection:** a forward disturbance step during steady following.

The team draws the system block diagram separately in SVG. Code does not generate it.

Figure format:

- matplotlib, saved as PDF and as PNG at 300 dpi to `figures/`
- 6.5 in wide; axis labels with units; no titles (captions go in the report)
- Consistent colors across all figures: true, measured, estimated, setpoint, command

## 9. Repository Layout

```
drone-sim/
  CLAUDE.md
  SPEC.md
  pyproject.toml
  config/
    params.yaml
    gains.yaml
  src/dronesim/
    params.py        # load and validate config into dataclasses
    plant.py         # nonlinear planar plant, RK4 step
    sensor.py        # camera model, noise, detection rules, delay queue
    link.py          # command link delay and zero-order hold
    estimator.py     # range Kalman filter (numpy + stdlib only)
    controllers.py   # yaw and range controllers (numpy + stdlib only)
    scenarios.py     # person trajectories S1 to S5
    sim.py           # multi-rate simulation loop, logging
    linear.py        # transfer functions, frequency response, margins
    metrics.py       # metrics from logged runs
    plotting.py      # shared style and color map
  scripts/
    tune_gains.py
    run_scenarios.py   # writes results/*.npz and results/metrics.csv
    make_figures.py    # reads results/, writes figures/
    sweep.py           # delay and sensitivity sweeps
  tests/
  results/
  figures/
```

## 10. Tests

- **Plant:** a constant pitch command settles forward speed at g·tan(θ)/k_v; a yaw-rate command step produces a heading ramp after the lag.
- **Sensor:** with zero noise, inversion returns true range and bearing; a person outside the field of view produces no measurement.
- **Delay:** a measurement becomes available exactly T_vision after capture; a command takes effect exactly T_link after it is issued.
- **Signs:** a person to the left produces a positive yaw-rate command; a person too far produces a positive pitch command.
- **Model agreement:** for small steps without noise, nonlinear and linear step responses agree within 5%.

## 11. Build Order (Today)

1. Config loading, plant, sensor, delay queue, link; tests for each.
2. Estimator, controllers, simulation loop; run S1 and S2.
3. Linear models and gain tuning; F2 and F3.
4. S3 to S5, metrics; F1, F4, F5.
5. Delay sweep; F6.
6. Should and Could figures if time allows.
7. Report.

## 12. Report Outline

1. Introduction: system goal and Phase 1 scope (simulation in place of the hardware build)
2. System architecture: shared-autonomy block diagram
3. Models and assumptions: plant, sensor, command link, parameter table
4. Controller and estimator design: structure, gain design, margins
5. Results: scenario responses and metrics table
6. Delay tolerance and parameter sensitivity
7. Limitations and hardware validation plan: webcam keypoint characterization, yaw test fixture, command link selection
