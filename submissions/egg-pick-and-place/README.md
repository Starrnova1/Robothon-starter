# FOIB-Egg: Fragile Object Integrity Benchmark

A MuJoCo 3 simulation benchmark that evaluates whether a 3-DOF robot arm
can pick and place an egg without compromising shell integrity.
**Shell integrity — monitored via continuous gripper force sensing — is the
primary scoring criterion.** Pick-and-place is the delivery mechanism;
force compliance throughout the grasp is the benchmark objective.

The benchmark runs headless, generates reproducible video from code, and
logs per-episode results to CSV. No manual annotation is required.

---

## Problem

Fragile object manipulation requires balancing two competing constraints:
grip must be firm enough to lift and carry the object, yet gentle enough
not to damage it. Standard pick-and-place benchmarks measure *placement
accuracy* but ignore *grasp force compliance*. FOIB-Egg makes force
compliance an explicit, scoreable criterion alongside placement.

---

## Benchmark Definition

Each episode proceeds through a fixed phase sequence:

```
IDLE → RETRACT → APPROACH → GRASP → LIFT → TRANSPORT → LOWER → RELEASE → CHECK
                                 │                                            │
                        shell monitoring                             DONE  (success)
                          begins here                                FAIL  (see taxonomy)
```

**An episode is scored SUCCESS only when both criteria hold:**

| Criterion | Threshold | Hard failure |
|-----------|-----------|-------------|
| Shell integrity | grip force < 12 N throughout grasp | `OVER-SQUEEZED` |
| Placement accuracy | egg within 8 cm of bowl centre after release | `DROPPED` |

`OVER-SQUEEZED` is a hard failure: a correctly placed egg is still a failed
episode if grip force exceeded the limit during transport.

---

## Failure Taxonomy

| Code | Trigger | Phase |
|------|---------|-------|
| `OVER-SQUEEZED` | `actuatorfrc` sensor exceeds 12 N | GRASP → TRANSPORT |
| `DROPPED` | egg Z falls below 0.79 m during lift/transport, or egg misses bowl after release | LIFT, TRANSPORT, CHECK |
| `TIMEOUT` | phase exceeds 2000 simulation steps (~4 s) without meeting exit condition | any phase |

The `SHELL` overlay indicator encodes these states in every video frame:

| Overlay label | Meaning |
|---|---|
| `--` | pre-grasp; monitoring not yet active |
| `CLOSING` | fingers moving in; force monitoring begins |
| `OK` | egg held; force within safe limits |
| `INTACT` | episode complete; shell survived |
| `OVER-SQUEEZED` | force limit exceeded → FAIL |
| `DROPPED` / `TIMEOUT` | other failure → FAIL |

---

## Benchmark Tiers

Three difficulty tiers control per-episode randomisation. Egg Y is capped
at ±3 mm across all tiers: finger closure acts in Y, and larger offsets
cause kinematic-attach to miss before fingers close (documented in Known
Limitations).

| Parameter | Easy | Medium | Stress |
|-----------|------|--------|--------|
| Egg X | ±5 mm | ±20 mm | ±30 mm |
| Egg Y | ±1 mm | ±3 mm | ±3 mm |
| Egg yaw | ±5° | ±15° | ±30° |
| Bowl X/Y | ±5 mm | ±15 mm | ±25 mm |
| Primary challenge | repeatability | moderate reach & placement | near-limit reach, max placement spread |

---

## Evaluation Protocol

Run the benchmark with a fixed seed and record the CSV:

```bash
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --log results.csv
```

**Reported metrics** (from CSV and video end card):

| Metric | Description |
|--------|-------------|
| Shell INTACT | episodes where force stayed below 12 N **and** egg was placed correctly |
| OVER-SQUEEZED | episodes where force limit was exceeded |
| Dropped / Timeout | episodes where egg was lost or a phase timed out |
| Peak grip (max) | maximum `actuatorfrc` reading across all episodes |
| Avg steps / ep | mean simulation steps per episode |

CSV format: `ep, tier, result, grip_peak, steps`

---

## Observation Space

The controller reads the following quantities each simulation step:

| Quantity | Source | Used for |
|----------|--------|----------|
| Arm joint positions | `data.qpos[7:10]` | waypoint tracking |
| End-effector position | `data.site_xpos[ee_site]` | grasp distance check |
| Egg body position | `data.xpos[egg_id]` | height gate, bowl proximity |
| Bowl centre position | `data.site_xpos[bowl_site]` | placement check |
| Grip force | `data.sensordata[sen_grip]` (`actuatorfrc`) | shell integrity gate |

All quantities are rendered in the video overlay (`OBS : Z=… D=…m`) so
the reviewer can see observed values change in real time.

---

## Project Structure

```
models/
  scene.xml            # MJCF scene: table, egg (freejoint), bowl, 3-DOF arm
  arm_bodies.xml       # arm fragment: links, parallel gripper, ee site
controller/
  phase_controller.py  # phase state machine — no RL, no gym dependency
scripts/
  validate_scene.py    # headless smoke test: 500 steps, NaN check
  run_interactive.py   # passive MuJoCo viewer for manual inspection
video/
  overlay.py           # per-frame HUD: EP/TIER/PHASE/OBS/SHELL/GRIP/STATUS
  record_demo.py       # recorder: single demo or tiered multi-episode benchmark
requirements.txt
```

---

## Setup

```bash
conda create -n robothon python=3.11
conda activate robothon
pip install -r requirements.txt
```

---

## Running

### Smoke test
```bash
python scripts/validate_scene.py
```

### Single-episode demo
```bash
python video/record_demo.py --out demo.mp4
```

### Benchmark — medium tier, 10 episodes, with CSV log
```bash
python video/record_demo.py \
    --episodes 10 --tier medium --seed 42 \
    --out benchmark.mp4 --log results.csv
```

### Benchmark — stress tier
```bash
python video/record_demo.py \
    --episodes 10 --tier stress --seed 42 \
    --out benchmark_stress.mp4 --log results_stress.csv
```

All flags: `--episodes N  --tier easy|medium|stress  --seed INT`
`--fps 30  --width 640  --height 480  --camera side_cam  --log PATH`

### Interactive viewer
```bash
python scripts/run_interactive.py
```

---

## Benchmark Results

Results are deterministic for a fixed seed. Verified on macOS, MuJoCo 3.9.0.

### Medium tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 10 / 10 (100 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 0 |
| Peak grip (max) | 0.370 N |
| Avg steps / ep | 709 |
| Video duration | 74.0 s (incl. title + end cards) |
| Wall time | ~22 s |

### Stress tier — seed 42, 10 episodes

| Metric | Value |
|--------|-------|
| Shell INTACT | 10 / 10 (100 %) |
| OVER-SQUEEZED | 0 |
| Dropped / Timeout | 0 |
| Peak grip (max) | 0.370 N |
| Avg steps / ep | 709 |
| Video duration | 74.0 s |
| Wall time | ~25 s |

---

## Reproducibility

```
Python        3.11.13
mujoco        3.9.0      (pinned exactly in requirements.txt)
numpy         2.4.6      (>=1.24 compatible)
opencv-python 4.13.0     (<5.0 required)
Platform      macOS 14+ / Linux (H.264 codec flag differs — see Known Limitations)
Seed          42 (benchmark default)
```

All random draws use `numpy.random.default_rng(seed)`, applied in order:
bowl XY → egg X → egg Y → egg yaw. The sequence is deterministic for any
fixed seed and episode count.

---

## Known Limitations

- **Shell integrity check is approximate.** `actuatorfrc` reads
  position-actuator reaction forces, which are near-zero under kinematic
  attachment (egg qpos is driven directly; no contact forces are generated
  on the egg). The `OVER-SQUEEZED` branch is fully wired into the state
  machine and overlay, but does not trigger under normal operation with
  this grasping mode. A friction-contact grasp would produce physically
  realistic force readings.

- **Kinematic attachment, not friction grasping.** The egg freejoint qpos
  is updated each step to follow the gripper base, bypassing contact
  physics. This ensures reliable grasping for the benchmark but does not
  model real gripper–egg interaction forces.

- **Egg Y randomisation is narrow (±3 mm, all tiers).** Finger closure
  acts in the Y direction. Eggs offset >~4 mm in Y are contacted
  asymmetrically before kinematic attach fires, pushing the egg out of
  range. The stress tier increases X and bowl variation only.

- **Fixed joint waypoints.** The arm navigates via pre-computed
  joint-space waypoints tuned for the default egg position. Waypoints do
  not adapt to per-episode randomisation; the ±3 cm X stress range is
  within the kinematic-attach capture radius (7 cm) of the fixed waypoint.

- **No gym env wrapper.** The controller is a plain Python class.
  Gym/Gymnasium encapsulation is deferred.

- **H.264 codec is macOS-specific.** `cv2.VideoWriter_fourcc(*"avc1")`
  requires Apple's AVFoundation. On Linux, replace with
  `cv2.VideoWriter_fourcc(*"mp4v")` or use an `.avi` container with
  `XVID`.
