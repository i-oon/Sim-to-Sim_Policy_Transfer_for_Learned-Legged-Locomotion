# Sim-to-Sim Policy Transfer for Learned Legged Locomotion
**Isaac Gym → MuJoCo | Command Switching / Transient Response Study**

---
Sim-to-real transfer for legged locomotion policies faces significant challenges due to differences in simulation environments and real-world hardware, including discrepancies in actuator dynamics, contact modeling, and real-time constraints.

Sim-to-sim transfer serves as a crucial intermediate step, enabling us to evaluate a learned policy’s ability to adapt to different simulation environments before real-world deployment. This helps identify issues such as mismatches in actuator stiffness, friction models, and solver behavior, which could lead to instability during transient behaviors like command switching.

---
This repository accompanies **Exam 2** of the Sim2Real Internship Candidate Exam form VISTEC. The objective is to analyze **sim-to-sim policy transfer mismatch** for learned legged locomotion policies under **contact-rich dynamics**, with focus on **transient responses induced by command switching**. A locomotion policy is trained in **Isaac Gym (Sim A)** and transferred **without retraining** to **MuJoCo (Sim B)**.

<p align="center">
    <img width=45% src="sources\play_isaacgym_1.gif">
    <img width=41% src="sources\deploy_mujoco_1.gif">
    </br> Played policy on Isaac Gym then deployed to Mujoco
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Findings Summary](#key-findings-summary)
- [Research Questions](#research-questions)
- [Robot Platform](#robot-platform)
- [Simulators & Configuration](#simulators--configuration)
- [Observation Space](#observation-space-48-dimensions)
- [Command Switching Scenarios](#command-switching-scenarios)
- [Evaluation Metrics](#evaluation-metrics)
- [Transient Response Analysis](#transient-response-analysis)
  - [S1 Stop Metrics](#s1-stop-transient-metrics-vx-06--00)
  - [S2 Turn Metrics](#s2-turn-transient-metrics-wz-00--10)
  - [S3 Lateral Metrics](#s3-lateral-transient-metrics-vy-03---03)
  - [Transient Summary](#transient-response-summary)
- [Experimental Results](#experimental-results)
  - [Stage 0: Sanity Checks](#stage-0-baseline-parity-sanity-checks)
  - [Stage 1: Baseline Performance](#stage-1-baseline-performance-steady-state--transient)
  - [Stage 1.5: Parameter Ablation](#stage-15-parameter-ablation-one-factor-at-a-time)
  - [Stage 2: Foot Friction Sweep](#stage-2-foot-friction-sweep)
  - [Stage 3: Observation Delay](#stage-3-observation-delay)
- [Summary of Mismatch Sources](#summary-of-mismatch-sources)
- [Conclusions](#conclusions)
- [Bonus: Mismatch Reduction](#bonus-mismatch-reduction-via-learned-actuator-models)
  - [Approach 1: ActuatorNet](#approach-1-actuatornet-direct-torque-prediction)
  - [Approach 2: Residual Learning](#approach-2-residual-learning-pd--learned-correction)
  - [Approach 3: Domain Randomization](#approach-3-domain-randomization)
  - [Final Comparison](#final-comparison-all-approaches)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Key Findings Summary

| Finding | Impact | Section |
|---------|--------|---------|
|**9.4% vx tracking gap** in steady-state | Baseline mismatch exists | [Stage 1](#stage-1-baseline-performance-steady-state--transient) |
| **20ms delay causes FALL** during turns | Critical for sim-to-real | [Stage 3](#stage-3-observation-delay) |
|**Kp=30 in MuJoCo ≈ Kp=20 in Isaac Gym** | 50% stiffness difference | [Stage 1.5](#stage-15-parameter-ablation-one-factor-at-a-time) |
|**Foot friction is bottleneck**, not floor | μ_foot=0.8 reduces pitch 88% | [Stage 2](#stage-2-foot-friction-sweep) |
|**Opposite yaw conventions** between sims | Sign difference in wz | [Stage 1](#stage-1-key-observation-divergent-yaw-behavior) |
|**RMA settling 31% faster** than PD | 220ms vs 320ms in S1 Stop | [Transient Analysis](#s1-stop-transient-metrics-vx-06--00) |
|**RMA reduces pitch 71%** in S1 Stop | 1.4° vs 4.8° | [Transient Analysis](#s1-stop-transient-metrics-vx-06--00) |
|**Residual 75% faster rise** in S2 Turn | 20ms vs 80ms | [Transient Analysis](#s2-turn-transient-metrics-wz-00--10) |
|**Data diversity > R² score** | ActuatorNet V2 (94.55%) beats V1 (99.21%) | [Bonus](#actuatornet-v2-policy-driven-excitation-data) |
|**Hwangbo excitation > policy-driven** | ActuatorNet V3 fixes V2's S2 instability | [Bonus](#actuatornet-v3-hwangbo-style-excitation-fixed-s2-instability) |
|**ActuatorNet V3 reduces S2 pitch 84%** | 17.8° → 2.8° | [Bonus](#actuatornet-v3-hwangbo-style-excitation-fixed-s2-instability) |
|**Residual Learning improves tracking 12-44%** | Best accuracy approach | [Bonus](#approach-2-residual-learning-pd--learned-correction) |
| **RMA reduces pitch/roll 22-31%** | Best stability approach | [Bonus](#approach-3-rma-rapid-motor-adaptation) |

---

## Overview

This repository accompanies **Exam 2** of the Sim2Real Internship Candidate Exam form VISTEC.  

The objective is to analyze **sim-to-sim policy transfer mismatch** for learned legged locomotion policies under **contact-rich dynamics**, with focus on **transient responses induced by command switching**.

A locomotion policy is trained in **Isaac Gym (Sim A)** and transferred **without retraining** to **MuJoCo (Sim B)**.

<p align="center">
    <img width=45% src="sources\play_isaaclab.gif">
    <img width=45% src="sources\play_isaacgym.gif">
    </br> Similarity in IsaacLab and Isaac Gym
</p>

While we have verified that Isaac Lab can produce deployable policies with equivalent MuJoCo performance, we selected **Isaac Gym as our primary framework** due to its simpler sim-to-sim pipeline of requiring no joint order remapping and its well-adoption in legged locomotion research.

---

## Research Questions

1. **Why do locomotion policies that exhibit similar steady-state performance diverge significantly during transient command switches?**

2. **Which sim-to-sim mismatches (actuator stiffness, solver dynamics, contact models) are amplified during high-acceleration maneuvers?**

3. **Can transient response metrics serve as a more sensitive indicator of sim-to-real transferability than steady-state tracking error?**

---

## Robot Platform

<p align="center">
    <img width=60% src="sources\Robots_Unitree_Go2.png">
    </br> The Unitree Go2 is a quadrupedal mobile robot designed for a variety of applications including research, field operations, education, and industrial tasks.
</p>


- **Robot:** Unitree Go2 Quadruped
- **DOF:** 12 (3 joints × 4 legs)
- **Joint Order Isaac Gym:** FL, FR, RL, RR (hip, thigh, calf per leg)
- **Joint Order MuJoCo qpos:** FL, FR, RL, RR (same as Isaac Gym)
- **Actuator Order MuJoCo ctrl:** FR, FL, RR, RL (requires remapping)


The Go2 uses **quasi-direct drive (QDD) actuators** with low gear ratio (~6-10:1), placing its dynamics complexity between platforms studied in prior sim-to-real work:

| Paper | Robot | Actuator Type | Modeling Approach |
|-------|-------|--------------|-------------------|
| Tan et al. (2018) | Minitaur | Direct-drive (no gears) | Analytical model (T = Kt·I) |
| Hwangbo et al. (2019) | ANYmal | Series-Elastic (SEA) | Learned ActuatorNet |
| Peng et al. (2018) | Laikago | Proprioceptive / QDD | Domain randomization |
| Kumar et al. (2021) | Unitree A1 | QDD  | RMA adaptation |

The Go2 (and its predecessor A1) sits in the **middle ground**: simple enough that basic PD control works for steady-state, but with sufficient non-linearities (gear friction, torque saturation) that might become problematic during transient maneuvers. 

This makes it an ideal testbed for comparing all three paradigms, analytical models (Tan), learned actuator models (Hwangbo), and adaptive policies (Kumar).


---

## Simulators & Configuration

### Sim A: Isaac Gym

| Parameter | Value |
|-----------|-------|
| Physics Engine | PhysX |
| sim.dt | 0.005s (200 Hz) |
| decimation | 4 |
| **Policy rate** | **50 Hz** (0.02s) |
| Kp (stiffness) | 20 N·m/rad |
| Kd (damping) | 0.5 N·m·s/rad |
| action_scale | 0.25 |
| Friction | 1.0 |
| Quaternion format | **(x, y, z, w)** |
| **Torque limits** | **±30 N·m** (implicit clipping) |

### Sim B: MuJoCo

| Parameter | Value |
|-----------|-------|
| Physics Engine | MuJoCo |
| timestep | 0.005s (200 Hz) |
| decimation | 4 |
| **Policy rate** | **50 Hz** (0.02s) |
| Kp (stiffness) | 20 N·m/rad |
| Kd (damping) | 0.5 N·m·s/rad |
| action_scale | 0.25 |
| Floor friction | 1.0 |
| Robot feet friction | 0.4 |
| Quaternion format | **(w, x, y, z)** |
| **Torque limits** | None (must add manually) |

---

## Observation Space (48 dimensions)

```
obs[0:3]   = base_lin_vel * 2.0        # Linear velocity (body frame)
obs[3:6]   = base_ang_vel * 0.25       # Angular velocity (body frame)
obs[6:9]   = projected_gravity         # Gravity in body frame
obs[9:12]  = cmd * cmd_scale           # Command [vx, vy, wz]
obs[12:24] = (joint_pos - default) * 1.0  # Joint positions
obs[24:36] = joint_vel * 0.05          # Joint velocities
obs[36:48] = last_action               # Previous action
```

**Critical:** Velocities must be in **body frame**, not world frame.

---

## Command Switching Scenarios

| Scenario | Before (t<3s) | After (t≥3s) | Test Focus |
|----------|---------------|--------------|------------|
| **S1 Stop** | vx=0.6, vy=0, wz=0 | vx=0, vy=0, wz=0 | Sudden stop |
| **S2 Turn** | vx=0.4, vy=0, wz=0 | vx=0.4, vy=0, wz=1.0 | Sudden turn |
| **S3 Lateral** | vx=0.3, vy=+0.3, wz=0 | vx=0.3, vy=-0.3, wz=0 | Direction flip |

### Visual Representation

**S1 Stop Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
     cmd = [0.6, 0, 0]    cmd = [0, 0, 0]
     (walk straight)       (stop instantly!)
```

**S2 Turn Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
     cmd = [0.4, 0, 0]    cmd = [0.4, 0, 1.0]
     (walk straight)       (turn instantly!)
```

**S3 Lateral Scenario:**
```
t=0s ─────────────► t=3s ─────────────────► t=6s
  cmd = [0.3, 0.3, 0]    cmd = [0.3, -0.3, 0]
   (walk diagonal)      (flip direction instantly!)
```

---

## Evaluation Metrics

### Steady-State Metrics
- Velocity tracking error (vx, vy, wz)
- RMSE

### Transient / Shock Metrics
- **Peak joint torque** — maximum |τ| during transient
- **Max roll/pitch** — body stability during command switch
- **Velocity overshoot** — how much velocity exceeds target
- **Fall detection** — roll/pitch > 1.0 rad or height < 0.15m

### Comparative Metrics
- Δ(peak torque) = MuJoCo − Isaac Gym
- Δ(max pitch) = MuJoCo − Isaac Gym

---

## Transient Response Analysis

This section provides detailed transient response metrics for command switching scenarios, measuring rise time, settling time, overshoot, and peak values.

### S1 Stop: Transient Metrics (vx: 0.6 → 0.0)


<p align="center">
    <img width=45% src="sources\mujoco_s1.gif">
    </br> S1 Scenerio of Straight Walk then Stop in Mujoco
</p>



| Metric | PD Only | PD + Residual | RMA | ActuatorNet V2 | Best |
|--------|---------|---------------|-----|----------------|------|
| **vx Rise time (10-90%)** | 260 ms | 200 ms | **160 ms** | 200 ms | RMA |
| **vx Settling time (5%)** | 320 ms | 280 ms | **220 ms** | 260 ms | RMA |
| **vx Overshoot** | 3.4% | **-0.3%** | 0.4% | -0.8% | Residual |
| **wz Settling time** | 920 ms | 320 ms | **260 ms** | 260 ms | RMA/ActNet |
| **Peak torque** | 15.62 N·m | 13.39 N·m | 12.50 N·m | **12.15 N·m** | ActuatorNet V2 |
| **Peak pitch** | 4.8° | 2.8° | **1.4°** | 4.6° | RMA |
| **Peak roll** | **1.6°** | 2.5° | 2.3° | 2.2° | PD |



**Key Observations:**
- RMA achieves **fastest settling** (220ms vs 320ms for PD) — **31% improvement**
- **ActuatorNet V2 has lowest peak torque** (12.15 N·m) — **22% reduction** from PD
- Residual & ActuatorNet V2 have **near-zero overshoot** (smoothest response)
- RMA has **best pitch stability** (1.4° vs 4.8°) — **71% reduction**
- ActuatorNet V2 performs well in S1 — comparable to RMA in settling time

---

### S2 Turn: Transient Metrics (wz: 0.0 → 1.0)

<p align="center">
    <img width=45% src="sources\mujoco_s2.gif">
    </br> S2 Scenerio of Straight Walk then Turn in Mujoco
</p>

| Metric | PD Only | PD + Residual | RMA | ActuatorNet V2 | Best |
|--------|---------|---------------|-----|----------------|------|
| **vx Rise time** | 40 ms | **20 ms** | 60 ms | 20 ms | Residual/ActNet |
| **wz Rise time** | 80 ms | **20 ms** | 160 ms | 320 ms | Residual |
| **wz Settling time** | 2980 ms | — | — | 2980 ms | — |
| **wz Overshoot** | -16.2% | **-10.0%** | -22.7% | 26.3% | Residual |
| **Peak torque** | 13.55 N·m | 13.51 N·m | **13.23 N·m** | 28.49 N·m X | RMA |
| **Peak pitch** | 4.6° | **3.8°** | 3.9° | 17.8° X | Residual |
| **Peak roll** | 5.1° | 6.1° | **2.4°** | 15.3° X | RMA |

**Key Observations:**
- Residual has **fastest wz rise time** (20ms vs 80ms) — **75% faster**
- Residual has **smallest wz overshoot** (-10% vs -16% to -23%)
- RMA has **best roll stability** (2.4° vs 5.1°) — **53% reduction**
- **ActuatorNet V2 is unstable in S2** — high torque (28.49 N·m), pitch (17.8°), roll (15.3°)
- Turn scenario reveals largest controller differences

---

### S3 Lateral: Transient Metrics (vy: +0.3 → -0.3)

<p align="center">
    <img width=45% src="sources\mujoco_s3.gif">
    </br> S3 Scenerio of Lateral Walking in Mujoco
</p>

| Metric | PD Only | PD + Residual | RMA | ActuatorNet V2 | Best |
|--------|---------|---------------|-----|----------------|------|
| **wz Rise time** | 260 ms | 340 ms | **220 ms** | 520 ms | RMA |
| **Peak torque** | 13.77 N·m | 13.76 N·m | **12.18 N·m** | 13.56 N·m | RMA |
| **Peak pitch** | 5.2° | 5.2° | **4.8°** | 5.1° | RMA |
| **Peak roll** | 3.6° | 5.7° | **2.1°** | 2.9° | RMA |

**Key Observations:**
- RMA consistently has **lowest peak torque** (12.18 N·m) — **12% reduction**
- RMA has **best roll stability** (2.1° vs 3.6°) — **42% reduction**
- ActuatorNet V2 performs well in S3 — similar to PD baseline
- Lateral scenario is less demanding than turn (S2)

---

### Transient Response Summary

| Controller | S1 Stop | S2 Turn | S3 Lateral | Overall |
|------------|---------|---------|------------|---------|
| **PD Only** | Baseline | Oscillatory | Baseline | Baseline |
| **PD + Residual** | Fast rise, smooth | **Fastest wz (20ms)** | Similar to PD | Best speed |
| **RMA** | **Best settling (220ms)** | **Best stability** | **Best stability** | **Best overall** |
| **ActuatorNet V2** | ✓ **Best torque** | ⚠️ **Unstable** | ✓ Good | Limited use |

**Recommendations:**
- **For best stability/robustness:** RMA Policy
- **For best tracking speed:** PD + Residual Learning
- **For ActuatorNet users:** Use **V3 (Hwangbo-style)**, avoid V2 for turn scenarios
- **Key V2→V3 improvement:** S2 Turn pitch reduced from 17.8° to 2.8° (84% improvement)

**Overall Findings:**

1. **RMA excels at stability metrics** — consistently lowest torque, pitch, and roll
2. **Residual excels at response speed** — fastest rise time in S2 Turn
3. **ActuatorNet V3 fixes V2's critical weakness** — now stable in S2 Turn scenario
4. **Trade-off exists:**
   - RMA: Best stability, higher complexity
   - Residual: Faster response, moderate complexity
   - ActuatorNet V3: Good balance, systematic data collection required
5. **Transient metrics reveal differences not visible in steady-state**


---

## Experimental Results

### Stage 0: Baseline Parity (Sanity Checks)

**Goal:** Ensure policy interface is correctly implemented before measuring physics mismatch.

| Check | Status | Result |
|-------|--------|--------|
| Zero-Action Stability | ✓ Pass | Height stable ~0.27m, roll/pitch < 3° |
| Observation Parity | ✓ Pass | Gravity diff < 0.03, lin_vel diff < 0.05 |
| Joint Order Verification | ✓ Pass | qpos order matches, ctrl order remapped |

**Key Implementation Fixes:**

1. **Quaternion Convention:**
   - Isaac Gym: (x, y, z, w)
   - MuJoCo: (w, x, y, z)

2. **Actuator Remapping:**
```python
# Isaac Gym: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
# MuJoCo:    FR(0-2), FL(3-5), RR(6-8), RL(9-11)
d.ctrl[0:3] = tau[3:6]   # FR
d.ctrl[3:6] = tau[0:3]   # FL
d.ctrl[6:9] = tau[9:12]  # RR
d.ctrl[9:12] = tau[6:9]  # RL
```

3. **Velocity Frame Transformation:**
```python
# MuJoCo gives world frame → transform to body frame
base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)
```

---

### Stage 1: Baseline Performance (Steady-State + Transient)

> Before attributing sim-to-sim mismatch to any specific source, we first establish a baseline comparison under nominal and well-controlled conditions. This step answers a fundamental question: *does a gap exist at all when both simulators are configured as similarly as possible?*

**Goal:** Measure sim-to-sim gap under nominal conditions (flat ground, default friction, no noise/delay) both using PD Controllers.

#### Steady-State Baseline (cmd: vx=0.5 m/s, 10 seconds)

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|-----|
| vx mean | 0.498 m/s | 0.451 m/s | **-0.047 (9.4%)** |
| vx RMSE | 0.079 | 0.066 | -0.013 |
| vy error | 0.011 | 0.010 | -0.001 |
| wz error | 0.216 | 0.007 | -0.209 |
| Torque mean | 2.56 N·m | 2.97 N·m | +0.41 |
| Torque max | 17.29 N·m | 15.71 N·m | -1.58 |

#### S1: Stop Shock (vx: 0.6 → 0.0)

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|---|
| Steady-State vx | 0.610 m/s | 0.566 m/s | -0.044 |
| Transient vx mean | 0.065 m/s | 0.040 m/s | -0.025 |
| Peak torque | 15.26 N·m | 16.29 N·m | +1.03 |
| Max pitch | 3.8° | 4.8° | +1.0° |
| Fallen | No | No | — |

#### S2: Turn Shock (wz: 0.0 → 1.0)

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|---|
| Steady-State vx | 0.407 m/s | 0.353 m/s | -0.054 |
| Transient wz mean | -0.178 rad/s* | 0.682 rad/s | — |
| Peak torque | 11.37 N·m | 15.32 N·m | **+3.95** |
| Max pitch | 3.0° | 5.0° | **+2.0°** |
| Fallen | No | No | — |

*Note: Isaac Gym wz has opposite sign convention — see below.

#### S3: Lateral Flip (vy: +0.3 → -0.3)

| Metric | Isaac Gym | MuJoCo | Gap |
|--------|-----------|--------|---|
| Steady-State vx | 0.327 m/s | 0.263 m/s | -0.064 |
| Transient vx mean | 0.298 m/s | 0.263 m/s | -0.035 |
| Peak torque | 12.75 N·m | 13.77 N·m | +1.02 |
| Max pitch | 3.8° | 5.2° | +1.4° |
| Fallen | No | No | — |


<p align="center">
   <img src="plots\all_scenarios_overview.png" width="400" alt="Alt Text">
   </br> S1, S2 and S3 Scenerio Comparison between Isaac Gym and Mujoco
</p>


#### Stage 1 Key Observation: Divergent Yaw Behavior

A significant behavioral difference was observed in S2 Turn:

| Period | Isaac Gym wz | MuJoCo wz | Note |
|--------|-------------|-----------|------|
| t=0-1s (init) | **-0.807 rad/s** | +0.018 rad/s | Isaac rotates without command! |
| t=1-3s (pre-switch) | -0.422 rad/s | -0.004 rad/s | Isaac still rotating |
| t=3-4s (post-switch) | -0.196 rad/s | **+0.710 rad/s** | Opposite directions |
| Total yaw change | **-117.3°** (CCW) | **+118.0°** (CW) | Same magnitude, opposite sign |

**Raw Data Ranges:**
- Isaac Gym wz: [-1.098, -0.032] rad/s — **always negative**
- MuJoCo wz: [-0.112, +0.839] rad/s — **mostly positive after switch**

> ⚠️ **Note on Plots:** In comparison plots, Isaac Gym wz sign is flipped (`-wz`) to visually align the turning behavior. The raw data shows opposite signs.


<p align="center">
   <img src="plots\S2_turn_comparison.png" width="400" alt="kp_ablation" >
   </br> S2 Scenerio Comparison between Isaac Gym and Mujoco
</p>  

*S2 Scenerio will be mainly used in others experiment due to its clear behavior of trasient state.

---

#### Root Cause Analysis: Heading Command Mode

We investigated why Isaac Gym shows rotation when command wz=0:

**Initial Observation:**
| Test | Command wz | Actual wz | Result |
|------|-----------|-----------|--------|
| Zero action (no policy) | 0 | **0.000** | ✓ Environment OK |
| Policy + set wz=0 directly | 0 | **-0.554** | X Unexpected rotation |

**Root Cause Discovery:**

The config uses `heading_command = True`:
```python
# In legged_robot.py post_physics_step:
if self.cfg.commands.heading_command:
    self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
```

**This means:**
- `commands[:, 2]` (wz) is **NOT a direct command** — it's **computed from heading error**
- `commands[:, 3]` is the **target heading** (in radians)
- Policy was trained to track **heading**, not angular velocity
- When we set `commands[:, 2] = 0` directly, it gets **overwritten** by the heading controller

**Verification:**

| Test | Method | wz (cmd=0) | Result |
|------|--------|------------|--------|
| X Wrong | Set `commands[:, 2] = 0` | -0.55 rad/s | wz overwritten |
| ✓ Correct | Set `commands[:, 3] = current_heading` | **-0.009 rad/s** | No rotation |

**Fixed Results (using heading command correctly):**

| Scenario | wz Before Switch | wz After Switch | Status |
|----------|------------------|-----------------|--------|
| S1 Stop | -0.003 rad/s | -0.036 rad/s | ✓ |
| S2 Turn | -0.009 rad/s | -0.778 rad/s | ✓ |
| S3 Lateral | -0.008 rad/s | +0.005 rad/s | ✓ |




**Key Lesson:**
- Always check `env_cfg.commands.heading_command` before setting commands
- If `heading_command = True`: set `commands[:, 3]` (target heading)
- If `heading_command = False`: set `commands[:, 2]` (angular velocity)
- This is **not a bug** — it's a different control mode that requires different command interface


**Interpretation:** Both robots rotate approximately the same magnitude (~118°) but in **opposite directions**. This indicates:
1. A **sign convention difference** between PhysX and MuJoCo for yaw rate
2. Isaac Gym exhibits large **initial transient** that decays, while MuJoCo remains stable until command switch
3. The policy behaves fundamentally differently in the two simulators despite identical network weights


> At this stage, it remains unclear which physical factors are responsible for the observed transient divergence.

---

### Stage 1.5: Parameter Ablation (One-Factor-at-a-Time)

> To identify the source of the transient mismatch observed in Stage 1, we next vary individual simulator parameters in MuJoCo while holding all others fixed. This one-factor-at-a-time ablation aims to isolate which mismatches materially affect performance.

**Goal:** Identify causal sources of mismatch by varying parameters individually in MuJoCo.

#### Kp (Stiffness) Sweep

| Kp | vx mean | vx error | Notes |
|----|---------|----------|-------|
| 10 | 0.101 m/s | 0.399 | Nearly crawling |
| **20 (baseline)** | **0.451 m/s** | **0.049** | Default |
| **30** | **0.466 m/s** | **0.034** | **Best match to Isaac Gym** |
| 40 | 0.407 m/s | 0.093 | Worse (overshoot) |


**Finding:** Kp=30 in MuJoCo best matches Isaac Gym (Kp=20) performance. MuJoCo requires ~50% higher stiffness to achieve equivalent actuator response.

<p align="center">
   <img src="plots\kp_ablation.png" width="400" alt="kp_ablation">
   </br> Kp=30 in Mujoco best matches Isaac Gym 
</p>

#### dt (Timestep) Sweep

| dt | decimation | Policy rate | vx mean | vx error |
|----|------------|-------------|---------|----------|
| 0.002 | 10 | 50 Hz | 0.410 | 0.090 |
| 0.005 | 4 | 50 Hz | 0.451 | 0.049 |
| 0.01 | 2 | 50 Hz | 0.408 | 0.092 |

**Finding:** Timestep has minimal effect when policy rate is kept constant at 50 Hz.

#### Floor Friction Sweep

| Floor friction | vx mean | vx error |
|----------------|---------|----------|
| 0.5 | 0.451 | 0.049 |
| 1.0 | 0.451 | 0.049 |
| 1.5 | 0.451 | 0.049 |

**Finding:** Floor friction has no measurable effect.

> The floor friction sweep initially produced a surprising result: varying μ_floor between 0.5 and 1.5 resulted in no observable change. This raised a critical follow-up question: Is friction truly irrelevant, or are we varying the wrong parameter?

---

### Stage 2: Foot Friction Sweep

The floor friction sweep produced no observable change. This motivated testing **foot friction** directly, since MuJoCo uses geometric mean of contacting surfaces:

$$\mu_{effective} \approx \sqrt{\mu_{floor} \times \mu_{foot}}$$

**Method:** Vary foot friction μ_foot ∈ {0.2, 0.4, 0.8} while holding floor friction constant at 1.0.

#### S2: Turn Shock (wz: 0.0 → 1.0)

| Metric | μ_foot=0.2 | μ_foot=0.4 (baseline) | μ_foot=0.8 |
|--------|------------|----------------------|------------|
| Steady-State vx | 0.294 m/s | 0.353 m/s | 0.353 m/s |
| wz overshoot | **+1.868 rad/s** | -0.161 rad/s | -0.254 rad/s |
| Max roll | **17.8°** | 5.1° | **5.3°** |
| Max pitch | **25.1°** | 5.0° | **3.0°** |
| Peak torque | **27.57 N·m** | 15.32 N·m | **13.65 N·m** |

<p align="center">
   <img src="plots\foot_friction_ablation.png" width="500" alt="foot_friction_ablation">
   <br> Foot Friction of 0.2, 0.4, 0.8 while holding floor friction constant at 1.0.
</p>  

#### Stage 2 Findings

1. **Foot friction has significant impact** while floor friction did not — confirming that foot friction is the bottleneck.

2. **High friction (μ=0.8) improves stability:**
   - Max roll: 5.3° (vs 17.8° at μ=0.2) — **70% reduction**
   - Max pitch: 3.0° (vs 25.1° at μ=0.2) — **88% reduction**
   - Peak torque: 13.65 N·m (vs 27.57 N·m) — **50% reduction**

3. **Low friction (μ=0.2) causes severe instability:**
   - wz overshoot +1.868 rad/s (nearly 2x target!)
   - Peak torque nearly doubles
   - Robot approaches fall conditions

> **Interpretation:** When foot friction is low, the system enters a **slip-prone regime** where lateral forces cannot be fully transmitted to the ground. The policy compensates by increasing torque output, but this leads to larger body excursions and potential instability.
>
> This confirms that **friction mismatch between Isaac Gym (μ=1.0) and MuJoCo (μ_foot=0.4) contributes to the observed sim-to-sim gap**.

---

### Stage 3: Observation Delay

> Thus far, all experiments assume perfect and instantaneous state feedback. While this isolates physics-related mismatch, real robots operate under sensing latency. We therefore consider whether transient sensitivity is further amplified by observation delays.

**Goal:** Evaluate sensitivity to sensing latency during transient command switches.

**Method:** Introduce observation delay of 0, 1, 2 policy steps (0, 20, 40 ms) in MuJoCo.

**Scenario:** S2 Turn (most challenging based on Stage 1 results)

#### Results

<p align="center">
   <img width=45% src="sources\mujoco_s2.gif">
   <img width=45% src="sources\delay_20ms.gif">
   <br> Policy with no Letency vs. Policy with 20ms Letency
</p>


| Metric | 0 ms | 20 ms | 40 ms |
|--------|------|-------|-------|
| Steady-State vx | 0.353 m/s | 0.330 m/s | 0.322 m/s |
| wz overshoot | -0.161 rad/s | **+1.348 rad/s** | +0.974 rad/s |
| Max roll | 5.1° | **167.8°** | 18.8° |
| Max pitch | 5.0° | **35.1°** | 28.4° |
| Peak torque | 15.32 N·m | **27.57 N·m (+80%)** | 26.21 N·m (+71%) |
| **Fallen** | No | **YES** 🔴 | No |

<p align="center">
   <img src="plots\delay_ablation.png" width="500" alt="delay_ablation">
   <br> Observation Delay of 0, 1, 2 policy steps
</p>

#### Stage 3 Key Finding: 20ms Delay Causes Fall

**Just 20ms of observation delay (1 policy step) causes the robot to fall**

This is a critical finding for sim-to-real transfer:
- Real robots typically have 10-50ms sensing latency
- Transient maneuvers are highly sensitive to delay
- Steady-state metrics do not predict this failure mode

> **Interpretation:** Observation delay causes the policy to act on outdated state information. During rapid transient maneuvers like turning, even small delays lead to:
> - **Over-correction:** Policy continues commanding turn after robot has already turned
> - **Oscillation:** Delayed feedback creates unstable control loop
> - **Fall:** At 20ms delay, oscillations exceed recovery limits
>
> Interestingly, 40ms delay does not cause fall — this suggests **chaotic sensitivity** where specific delay values interact with gait timing in unpredictable ways.

---

## Summary of Mismatch Sources

| Source | Effect on Tracking | Effect on Stability | Recommendation |
|--------|-------------------|---------------------|----------------|
| **Kp (stiffness)** | High | Medium | Use Kp=30 in MuJoCo to match Isaac Gym Kp=20 |
| **Foot friction** | Medium | **High** | Match μ_foot to real robot (~0.6-0.8 for rubber) |
| **Floor friction** | None | None | Not a significant factor |
| **Timestep** | Low | Low | Keep policy rate constant |
| **Observation delay** | Medium | **Critical** | Add 20-40ms delay in sim for robustness |
| **Yaw convention** | — | — | Document and handle sign difference |
| **Torque clipping** | Medium | High | Isaac Gym clips at ±30 N·m (implicit) |

---

## Conclusions

### Key Findings

1. **Transient command switching reveals mismatch not visible in steady-state.** 
   - S2 Turn shows largest gaps in torque and stability metrics
   - Steady-state tracking differs by only ~9%, but transient behavior diverges significantly

2. **Simulators have opposite yaw conventions.**
   - Isaac Gym and MuJoCo rotate in opposite directions for same wz command
   - Both rotate ~118° magnitude, confirming policy works but conventions differ

3. **Actuator stiffness (Kp) requires ~50% increase in MuJoCo.**
   - Kp=30 in MuJoCo ≈ Kp=20 in Isaac Gym
   - Likely due to different contact solver dynamics

4. **Foot friction is the contact bottleneck, not floor friction.**
   - μ_foot=0.8 reduces peak torque by 50% and pitch by 88%
   - MuJoCo uses geometric mean, so lower value dominates

5. **20ms observation delay causes fall during turning.**
   - Critical finding for sim-to-real transfer
   - Transient maneuvers are highly sensitive to latency
   - Steady-state metrics do not predict this failure

6. **Isaac Gym applies implicit torque clipping at ±30 N·m.**
   - Discovered during residual learning analysis
   - This affects high-torque maneuvers like turning

### Answers to Research Questions

**Q1: Why do policies diverge during transients?**
> Contact dynamics (friction, solver) and observation timing affect transient response more than steady-state. The policy's corrective actions are amplified differently in each simulator. Additionally, Isaac Gym's implicit torque clipping is not present in MuJoCo by default.

**Q2: Which mismatches are amplified during high-acceleration maneuvers?**
> Foot friction and observation delay have the largest impact during transients. Actuator stiffness affects both steady-state and transient equally. Torque clipping becomes critical during high-demand maneuvers.

**Q3: Can transient metrics serve as better transferability indicators?**
> **Yes.** The 20ms delay causes fall during S2 Turn but has minimal effect on steady-state tracking. Transient metrics (peak torque, max pitch, fall detection) are more sensitive indicators.

### Recommendations for Sim-to-Real Transfer

1. **Tune Kp/Kd gains** — MuJoCo requires ~50% higher Kp than Isaac Gym
2. **Match foot friction** to real robot contact properties (rubber ≈ 0.6-0.8)
3. **Add observation delay** (20-40ms) during training/validation
4. **Test turning maneuvers** — they expose the largest gaps
5. **Verify sign conventions** for angular velocities and quaternions
6. **Use transient metrics** (peak torque, max pitch) in addition to tracking error
7. **Add torque clipping** in MuJoCo to match Isaac Gym behavior (±30 N·m)

---

## Bonus: Mismatch Reduction via Learned Actuator Models

This section explores using neural network-based actuator models to reduce sim-to-sim mismatch.

### Approach 1: ActuatorNet (Direct Torque Prediction)

**Concept:** Replace PD control entirely with a learned model that predicts torque from Isaac Gym data.

**Reference:** [actuator_net](https://github.com/sunzhon/actuator_net) — originally designed for sim-to-real transfer.

---

#### ActuatorNet V1: Normal Walking Data (Failed)

**Method:**
1. Collected 30,000 timesteps × 12 motors = 360,000 samples from **normal walking only**
2. Data format: input `[pos_error, pos_error_t-1, pos_error_t-2, vel, vel_t-1, vel_t-2]` → output `torque`
3. Trained MLP with R² = 99.21% on test set
4. Deployed in MuJoCo replacing PD control with `τ = ActuatorNet(state)`

<p align="center">
<img src="plots\actuator_net.png" width="400" alt="Alt Text">
<br> actuator_net's UI and Usage Example
</p>


**Results V1:**

| Metric | ActuatorNet V1 | PD Control |
|--------|----------------|------------|
| vx error | 0.277 m/s | **0.049 m/s** |
| Torque max | 26.2 N·m | **15.7 N·m** |

**Problem Analysis:**
- High R² (99.21%) but poor deployment performance
- **Cause: Overfit to normal walking dynamics** as data didn't cover extreme cases
- Training data ranges: vel ±20 rad/s, torque ±26 N·m (below limits)

---
#### ActuatorNet V2: Policy-Driven Excitation Data

Following the need for diverse data, we collected **policy-driven excitation trajectories**:

**Excitation Data Collection:**
```
Phase 1: Normal walking (varied commands)     - 300,000 samples
Phase 2: Random joint perturbations           - 150,000 samples  
Phase 3: High-speed commands                  - 150,000 samples
Phase 4: Sudden command switches              - 150,000 samples
─────────────────────────────────────────────────────────────
Total: 750,000 samples (25x more than V1)
```

**Data Coverage Improvement:**

| Metric | V1 (Normal) | V2 (Excitation) | Improvement |
|--------|-------------|-----------------|-------------|
| Samples | 30,000 | **750,000** | **25x** |
| Velocity range | ±20 rad/s | **±37 rad/s** | 1.85x |
| Torque range | ±26 N·m | **±35.5 N·m** | Reaches limits |

**Training Results:**
- Architecture: MLP [100 units, 4 layers] with softsign activation
- Features: `[pos_error, velocity, pos_error × velocity]`
- R² = 94.55% (lower than V1, but on diverse data = better generalization)

**Results V2 - S2 Command Switch (wz: 0→1.0 at t=3s):**

<p align="center">
    <img width=50% src="sources\actuator_netv2_s2.gif">
    <br> ActuatorNet V2 with S2 Command Switch
</p>


| Metric | PD Control | ActuatorNet V2 | Change |
|--------|------------|----------------|--------|
| Status | Stable | **Unstable** | — |
| Torque max | **15.32 N·m** | 28.49 N·m | +86% X |
| Max pitch | **5.0°** | 17.8° | +256% X |
| Max roll | **5.1°** | 15.3° | +200% X |

**V2 Limitation:** While V2 improved over V1, it still struggled with **transient turn commands (S2)** due to policy-driven data not covering the full actuator dynamics space.

---

#### ActuatorNet V3: Hwangbo-Style Excitation (Fixed S2 Instability)

**Reference:** Hwangbo et al., "[Learning agile and dynamic motor skills for legged robots](https://www.science.org/doi/10.1126/scirobotics.aau5872)" (Science Robotics 2019) 

The key insight from Hwangbo's paper is that **actuator data should be collected independently from locomotion policy**, using systematic excitation signals that cover the full operating range.

**Key Differences from V2:**

| Aspect | V2 (Policy-Driven) | V3 (Hwangbo-Style) |
|--------|-------------------|-------------------|
| Data source | Policy locomotion | **Systematic excitation** |
| Sinusoidal sweep | X None | ✓ 0.5-10 Hz |
| Chirp (freq sweep) | X None | ✓ 0.5→10 Hz |
| Torque saturation probing | Partial | ✓ Deliberate |
| Decoupled from task | X No | ✓ Yes |
| Identifiability | Low | **High** |

**Hwangbo Excitation Data Collection:**
```
Phase 1: Low-frequency sinusoids (0.5-2 Hz)   - 60,000 samples
Phase 2: Mid-frequency sinusoids (2-5 Hz)    - 60,000 samples
Phase 3: High-frequency sinusoids (5-10 Hz)  - 60,000 samples
Phase 4: Chirp sweep (0.5→10 Hz)             - 120,000 samples
Phase 5: Torque saturation probing           - 60,000 samples
Phase 6: Multi-sine (mixed frequencies)      - 60,000 samples
─────────────────────────────────────────────────────────────
Total: 420,000 samples with full frequency coverage
```

**Simulated Actuator Dynamics:**

To make the learning problem non-trivial (avoid R²=1.0 data leakage), we added realistic actuator dynamics:

```python
class ActuatorDynamics:
    motor_time_constant = 0.02s    # First-order lag
    viscous_friction = 0.1 N·m/(rad/s)
    coulomb_friction = 0.5 N·m
    noise_std = 0.1 N·m
```

**Training Results:**
- Architecture: MLP [100 units, 4 layers] with softsign activation
- Features: `[pos_error, pos_error_t-1, pos_error_t-2, vel, vel_t-1, vel_t-2]` (6 features with history)
- R² = 99.80% on test set
- Training stopped early at epoch 44

**Results V3 - S1 Stop (vx: 0.6 → 0.0):**

| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Rise time | 200 ms | **180 ms** | -10% ✓ |
| Settling time | 260 ms | **240 ms** | -8% ✓ |
| Peak torque | **12.15 N·m** | 14.03 N·m | +15% |
| Peak pitch | 4.6° | **1.5°** | **-67%** ✓ |
| Peak roll | 2.2° | 2.5° | +14% |

**Results V3 - S2 Turn (wz: 0.0 → 1.0):**
<p align="center">
   <img width=45% src="sources\actuator_netv2_s2.gif">
   <img width=45% src="sources\actuator_netv3_s2.gif">
   <br> ActuatorNet V2 vs. ActuatorNet V3
</p > 


| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Status | **Unstable** | ✓ **Stable** | **Fixed!** |
| Peak torque | 28.49 N·m X | **13.34 N·m** | **-53%** ✓ |
| Peak pitch | 17.8° X | **2.8°** | **-84%** ✓ |
| Peak roll | 15.3° X | **5.1°** | **-67%** ✓ |
| wz Rise time | 320 ms | **80 ms** | -75% ✓ |

**Results V3 - S3 Lateral (vy: +0.3 → -0.3):**

| Metric | V2 | V3 | Change |
|--------|-----|-----|--------|
| Peak torque | **13.56 N·m** | 13.88 N·m | +2% |
| Peak pitch | 5.1° | **4.5°** | -12% X |
| Peak roll | **2.9°** | 6.3° | +117% |

**V3 Key Achievement:** Successfully fixed the S2 Turn instability that plagued V2, with **84% reduction in peak pitch** and **53% reduction in peak torque**.

---

#### ActuatorNet Conclusions

| Version | Data Type | Normal Walking | Constant Turn | Command Switch (S2) |
|---------|-----------|----------------|---------------|---------------------|
| **V1** | Normal walking | X Worse | — | — |
| **V2** | Policy excitation | ✓ Comparable | ✓ Stable | ⚠️ **Unstable** |
| **V3** | Hwangbo excitation | ✓ Comparable | ✓ Stable | ✓ **Stable** |

**Key Insights:**

1. **Data diversity matters more than R²** — V1 had 99.21% R² but failed; V2 had 94.55% but works better
2. **Systematic excitation > policy-driven excitation** — V3's Hwangbo-style data collection fixed V2's S2 instability
3. **Frequency coverage is critical** — sinusoidal sweeps and chirp signals ensure full actuator dynamics coverage
4. **History features help** — V3 uses `[pos_error_t-2:t, vel_t-2:t]` for temporal modeling

**Recommendation:** For ActuatorNet to work reliably:
- Use **Hwangbo-style systematic excitation** (not policy-driven)
- Include **sinusoidal sweeps** across 0.5-10 Hz frequency range
- Add **chirp signals** for continuous frequency coverage
- Include **torque saturation probing** to reach actuator limits
- For best overall robustness, prefer **RMA** over ActuatorNet

---

### Approach 2: Residual Learning (PD + Learned Correction)

**Concept:** Instead of replacing PD control, learn a residual correction term:

```
τ_mujoco = τ_pd + Δτ_learned
```

Where `Δτ_learned` compensates for the difference between Isaac Gym and MuJoCo actuator responses.

**Method:**
1. Collected residual data: `Δτ = τ_isaac - τ_pd` from Isaac Gym
2. Residual statistics: mean=0.011 N·m, std=0.810 N·m, range=[-19.7, +18.0] N·m
3. Trained ResidualNet: input `[pos_error, velocity]` → output `Δτ`
4. Test RMSE: 0.577 N·m
5. Deployed: `τ_mujoco = τ_pd + Δτ_learned`

**Results - Steady-State Baseline (vx=0.5 m/s):**

| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| vx error | 0.049 m/s | 0.043 m/s | **-12%** ✓ |
| Torque max | 15.71 N·m | 15.42 N·m | -2% |

#### For futhermore and esier behavior comparison, we tested more on Continuous Turn Command of  (0.4, 0.0, 1.0)

**Results - Turn Command (wz=1.0 rad/s, constant):**

<p align="center">
    <img width=45% src="sources\pd_continuous_turn.gif">
    <img width=45% src="sources\pd_with_residual_continuous_turn.gif">
    <br> Controller vs. PD + Residual Learning Controller
</p>


| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| vx error | 0.093 m/s | 0.052 m/s | **-44%** ✓ |
| wz error | 0.395 rad/s | 0.338 rad/s | **-14%** ✓ |
| Status | **FALLEN** | Stable | **Fixed!** ✓ |

**Results - S2 Command Switch (wz: 0→1.0 at t=3s):**

| Metric | PD Only | PD + Residual | Improvement |
|--------|---------|---------------|-------------|
| Torque max | 15.32 N·m | 14.64 N·m | **-4.4%** ✓ |
| Max pitch | 5.0° | 4.4° | **-12%** ✓ |

**Conclusion:** Residual learning provides:
- **12-44% improvement** in velocity tracking
- **4-12% reduction** in peak torque and pitch excursion
- **Stabilization** of challenging maneuvers (turn command)

---

### Analysis: What Does Residual Learning Actually Learn?

Upon investigation, we discovered **why** residual learning works:

**Torque Analysis:**

| Metric | τ_pd (computed) | τ_isaac (actual) |
|--------|-----------------|------------------|
| Range | -46.6 to +29.1 N·m | **-28.6 to +30.0 N·m** |

**Key Finding:** Isaac Gym applies **torque clipping** at ~30 N·m limits

```python
# Isaac Gym internal implementation
torques = p_gains * (target - current) - d_gains * velocity
return torch.clip(torques, -torque_limits, +torque_limits)
```

**Residual Statistics by Torque Magnitude:**

| Condition | Sample Count | Mean Residual |
|-----------|--------------|---------------|
| \|τ_pd\| > 20 N·m | 73 | 0.242 N·m |
| \|τ_pd\| < 10 N·m | 175,630 | 0.014 N·m |

**But clipping alone is NOT sufficient:**

We tested explicit torque clipping:
```python
tau = np.clip(tau_pd, -30, +30)
```

| Approach | Turn Command Status |
|----------|---------------------|
| PD Only | **FALLEN** |
| PD + Clipping (±30) | **FALLEN** |
| PD + Residual | **Stable** ✓ |

**Deeper Analysis - Velocity-Dependent Compensation:**

| Velocity Range | Mean Residual | Interpretation |
|----------------|---------------|----------------|
| \|vel\| < 2 rad/s | -0.061 N·m | Negligible |
| \|vel\| 2-5 rad/s | **+0.345 N·m** | Positive compensation |
| \|vel\| 5-10 rad/s | **+1.462 N·m** | Large positive compensation |

**Key Correlations:**
- `velocity vs residual`: **-0.49** (strong negative correlation)
- `pos_error vs residual`: -0.31

**What Residual Learning Actually Learns:**

1. **Torque Clipping** — when |τ| > 30 N·m
2. **Velocity-Dependent Compensation** — when |vel| > 2 rad/s

```
Approximate learned function:
Δτ ≈ clipping_effect + velocity_compensation

Where velocity_compensation ≈ +0.3 to +1.5 N·m when |vel| > 2 rad/s
```

**Why This Matters:**
- During turns, joint velocities increase significantly
- Isaac Gym has implicit velocity-dependent dynamics (possibly friction compensation)
- Simple clipping doesn't capture this — it only limits magnitude
- Residual learning captures **both** clipping AND velocity compensation

---

### Approach 3: Domain Randomization

**Concept:** Train policy with domain randomization (DR) and privileged critic to learn robust behaviors that generalize across simulator differences.

**Reference:** Kumar et al., "[RMA: Rapid Motor Adaptation for Legged Robots](https://ashish-kmr.github.io/rma-legged-robots/)" (CoRL 2021) 

This concept draws direct inspiration from Phase 1 (Base Policy Training) of Kumar's paper, which proposes a method for constructing robust policies through training in highly diverse environments.

**Method:**
1. Created `go2_rma` environment with:
   - Domain randomization: friction [0.3, 1.5], mass [-1kg, +2kg], Kp/Kd [0.8x, 1.2x]
   - Privileged observations for critic (env params + ground truth velocities)
   - Actor uses only regular observations (no privileged info)

2. Training configuration:
   - Asymmetric actor-critic (actor: 48 obs, critic: 58 obs)
   - 5000 iterations
   - Final reward: 33.77, tracking accuracy: 94-95%

3. Deployment: Use actor policy only in MuJoCo

**Results - Baseline (vx=0.5 m/s):**

| Metric | Original Policy | RMA Policy | Change |
|--------|-----------------|------------|--------|
| vx error | **0.049 m/s** | 0.072 m/s | +47% X |
| Torque max | 15.71 N·m | **13.01 N·m** | **-17%** ✓ |

#### For futhermore and esier behavior comparison, we tested more on Continuous Turn Command of  (0.4, 0.0, 1.0)

**Results - Turn Command (wz=1.0 rad/s, constant):**

<p align="center">
    <img width=45% src="sources\pd_continuous_turn.gif">
    <img width=45% src="sources\randomization_policy.gif">
    Original Policy vs. Policy with Domain Randomization
</p>

| Metric | Original Policy | DR Policy | Change |
|--------|-----------------|------------|--------|
| Status | **FALLEN** | **Likely Stable** | **Fixed!** ✓ |
| vx error | N/A | 0.033 m/s | — |
| Torque max | exploded | 14.36 N·m | — |

**Results - S2 Command Switch (wz: 0→1.0 at t=3s):**

| Metric | Original Policy | DR Policy | Change |
|--------|-----------------|------------|--------|
| Torque max | 15.32 N·m | **13.23 N·m** | **-14%** ✓ |
| Max pitch | 5.0° | **3.9°** | **-22%** ✓ |
| Max roll | 5.1° | **3.5°** | **-31%** ✓ |
| vx error (after) | 0.040 m/s | **0.029 m/s** | **-28%** ✓ |

**Conclusion:** RMA provides:
- **Better stability** during challenging maneuvers (turn, command switch)
- **Lower peak torques** (-14% to -17%)
- **Reduced body excursions** (pitch -22%, roll -31%)
- Trade-off: Slightly worse steady-state tracking in baseline conditions

**Key Insight:** Domain randomization during training makes the policy robust to simulator differences. The policy learns to handle variations in friction, mass, and gains — which implicitly covers MuJoCo's different dynamics.

---

### Final Comparison: All Approaches

| Approach | Baseline vx | Turn Const | S2 Switch | S2 Pitch | S2 Roll | Complexity |
|----------|-------------|------------|-----------|----------|---------|------------|
| **PD Only** | 0.049 m/s | X **FELL** | ✓ Stable | 5.0° | 5.1° | Low |
| **PD + Clipping** | — | X **FELL** | — | — | — | Low |
| **ActuatorNet V1** | 0.277 m/s | — | — | — | — | Medium |
| **ActuatorNet V2** | 0.062 m/s | ✓ Stable | **Unstable** | 17.8° X | 15.3° X | Medium |
| **ActuatorNet V3** | ~0.06 m/s | ✓ Stable | ✓ **Stable** | **2.8°** ✓ | 5.1° | Medium |
| **PD + Residual** | **0.043 m/s** | ✓ Stable | ✓ Stable | 4.4° | 6.1° | Medium |
| **DR Policy** | 0.072 m/s | ✓ Stable | ✓ Stable | 3.9° | **2.4°** | High |

**Key Improvement V2 → V3:**
- S2 Turn: Pitch **17.8° → 2.8°** (84% reduction)
- S2 Turn: Torque **28.49 → 13.34 N·m** (53% reduction)
- S2 Turn: Status **Unstable → Stable**

**Recommendations:**
- **For best tracking accuracy:** PD + Residual Learning
- **For best stability/robustness:** RMA Policy  
- **For ActuatorNet users:** Use V3 (Hwangbo-style excitation), avoid V2 for transient scenarios
- **Avoid:** ActuatorNet V1 (normal data only) for any sim-to-sim transfer

---

## Usage

### Prerequisites

**1. Clone repositories:**
```bash
git clone https://github.com/

# MuJoCo robot models
git clone https://github.com/unitreerobotics/unitree_mujoco.git

# ActuatorNet
git clone https://github.com/sunzhon/actuator_net.git
```

**2. Create conda environment:**
```bash
conda create -n unitree_rl python=3.8 -y
conda activate unitree_rl
```

**3. Install dependencies:**
```bash
cd ~/6619_ws/unitree_rl_gym

# Isaac Gym (download from NVIDIA)
pip install -e isaacgym/python

# Core packages
pip install torch torchvision
pip install mujoco mujoco-viewer
pip install numpy scipy matplotlib pyyaml

# RL training
pip install -e .

# ActuatorNet
cd ~/6619_ws/actuator_net
pip install -e .
```

**4. Verify setup:**
```bash
conda activate unitree_rl
cd ~/6619_ws/unitree_rl_gym

# Test Isaac Gym
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"

# Test MuJoCo
python -c "import mujoco; print('MuJoCo OK')"
```

### Training

```bash
# Standard policy
python legged_gym/scripts/train.py --task=go2 --headless --max_iterations=5000

# RMA policy (with domain randomization)
python legged_gym/scripts/train.py --task=go2_rma --headless --max_iterations=5000
```

### Isaac Gym Evaluation

```bash
# Playback
python legged_gym/scripts/play.py --task=go2 --num_envs=1

# Baseline logging
python legged_gym/scripts/play_logging.py --task=go2

# Command switching
python legged_gym/scripts/play_cmd_switch.py --task=go2 --scenario S2_turn
```

### MuJoCo Deployment

```bash
# Basic deployment (interactive viewer)
python deploy/deploy_mujoco/deploy_mujoco_go2.py go2.yaml

# With metric logging
python deploy/deploy_mujoco/deploy_mujoco_go2_logging.py go2.yaml --duration 10

# Command switching scenarios
python deploy/deploy_mujoco/deploy_mujoco_go2_cmd_switch.py go2.yaml --scenario S2_turn

# Observation delay test
python deploy/deploy_mujoco/deploy_mujoco_go2_delay.py go2.yaml --scenario S2_turn --delay 1
```

### Mismatch Reduction Controllers

```bash
# PD + Residual Learning
python deploy/deploy_mujoco/deploy_mujoco_go2_residual.py go2.yaml --duration 10 --cmd 0.5 0.0 0.0

# RMA Policy
python deploy/deploy_mujoco/deploy_rma_cmd_switch.py go2_rma.yaml

# ActuatorNet V3 transient analysis (S1/S2/S3)
python deploy/deploy_mujoco/deploy_transient_actuator_net_v3.py S2 --headless
```

### Data Collection & Training (Bonus)

```bash
# Collect actuator data (Hwangbo-style excitation)
python legged_gym/scripts/collect_hwangbo_excitation.py --task=go2

# Collect residual data
python legged_gym/scripts/collect_residual_data.py --task=go2

# Train residual network
python legged_gym/scripts/train_residual_net.py
```

### Generate Plots

```bash
python scripts/plot_results.py
```

---

## Project Structure

```
unitree_rl_gym/
│
├── legged_gym/
│   ├── envs/go2/
│   │   ├── go2_config.py                              # Base Go2 configuration
│   │   ├── go2_rma_config.py                          # RMA config (domain randomization)
│   │   └── go2_rma_env.py                             # RMA env (privileged observations)
│   │
│   └── scripts/
│       ├── train.py                                   # Policy training
│       ├── play.py / play_logging.py                  # Playback & logging
│       ├── play_cmd_switch.py                         # Command switching in Isaac Gym
│       ├── collect_actuator_data_for_actuator_net.py  # ActuatorNet V1 data
│       ├── collect_excitation_data.py                 # Policy-driven excitation (V2)
│       ├── collect_hwangbo_excitation.py              # Hwangbo-style excitation (V3)
│       ├── collect_residual_data.py                   # Residual Δτ data
│       ├── train_actuator_net_v2.py                   # Train ActuatorNet V2
│       └── train_residual_net.py                      # Train residual network
│
├── deploy/deploy_mujoco/
│   ├── configs/
│   │   ├── go2.yaml                                   # Base MuJoCo config
│   │   ├── go2_rma.yaml                               # RMA policy config
│   │   ├── go2_kp_*.yaml                              # Kp ablation configs
│   │   └── go2_foot_*.yaml                            # Foot friction ablation configs
│   │
│   ├── deploy_mujoco_go2.py                           # Basic deployment
│   ├── deploy_mujoco_go2_logging.py                   # With metric logging
│   ├── deploy_mujoco_go2_cmd_switch.py                # Command switching
│   ├── deploy_mujoco_go2_delay.py                     # Observation delay test
│   ├── deploy_mujoco_go2_residual.py                  # PD + Residual Learning
│   ├── deploy_mujoco_go2_clipping.py                  # Explicit torque clipping
│   ├── deploy_mujoco_go2_actuator_net.py              # ActuatorNet V1
│   ├── deploy_mujoco_go2_actuator_net_v2.py           # ActuatorNet V2
│   ├── deploy_transient_actuator_net_v2.py            # ActuatorNet V2 transient
│   ├── deploy_transient_actuator_net_v3.py            # ActuatorNet V3 transient
│   ├── deploy_transient_analysis.py                   # General transient metrics
│   ├── deploy_rma_cmd_switch.py                       # RMA command switching
│   └── sanity_check_*.py                              # Validation scripts
│
├── unitree_mujoco/unitree_robots/go2/
│   ├── scene_flat.xml                                 # Flat terrain
│   ├── scene_foot_02.xml                              # μ_foot = 0.2
│   └── scene_foot_08.xml                              # μ_foot = 0.8
│
├── scripts/
│   └── plot_results.py                                # Generate comparison plots
│
├── logs/
│   ├── rough_go2/Jan30_23-48-03_/model_5000.pt       # Trained standard policy
│   ├── go2_rma/Jan31_20-18-52_/model_5000.pt         # Trained RMA policy
│   ├── sim2sim/                                       # Baseline & ablation logs
│   ├── transient_analysis/                            # Transient response logs
│   ├── residual_net.pt                                # Trained residual model
│   ├── residual_scaler.pkl                            # Residual feature scaler
│   ├── actuator_net_v2.pt                             # Trained ActuatorNet V2
│   ├── actuator_net_v2_scaler_X.pkl                   # V2 input scaler
│   └── actuator_net_v2_scaler_y.pkl                   # V2 output scaler
│
├── plots/                                             # Generated figures
└── README_SIM2SIM.md
```

**External dependency:**
```
~/actuator_net/app/resources/
├── hwangbo_excitation_data.csv                        # Hwangbo excitation data (V3)
├── actuator.pth                                       # Trained ActuatorNet V3 model
├── scaler.pkl                                         # V3 input scaler
└── motor_data.pkl                                     # Processed training data
```

---

## File Paths

| Description | Path |
|-------------|------|
| Original trained model | `logs/rough_go2/Jan30_23-48-03_/model_5000.pt` |
| Original exported policy | `logs/rough_go2/exported/policies/policy_1.pt` |
| RMA trained model | `logs/go2_rma/Jan31_20-18-52_/model_5000.pt` |
| RMA exported policy | `logs/go2_rma/exported/policies/policy_1.pt` |
| MuJoCo Go2 model | `unitree_mujoco/unitree_robots/go2/scene_flat.xml` |
| Residual network | `logs/residual_net.pt` |
| Residual scaler | `logs/residual_scaler.pkl` |
| Baseline logs | `logs/sim2sim/` |
| Command switch logs | `logs/sim2sim/cmd_switch/` |
| Delay logs | `logs/sim2sim/delay/` |
| Plots | `plots/` |

---

## Experimental Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Isaac Gym)                         │
│  train.py → model_5000.pt → export → policy_1.pt               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 0: SANITY CHECKS                          │
│  Zero-action stability · Observation parity · Joint ordering    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 1: BASELINE COMPARISON                    │
│  Isaac Gym ←→ MuJoCo | Scenarios: S1 Stop, S2 Turn, S3 Lateral │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 1.5: PARAMETER ABLATION (One Factor at a Time)     │
│  Kp sweep · dt sweep · Floor friction sweep                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: FOOT FRICTION SWEEP                       │
│  μ_foot: 0.2, 0.4, 0.8 | Finding: μ_foot=0.8 reduces pitch 88%│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: OBSERVATION DELAY                         │
│  0 / 20 / 40 ms | Finding: 20ms causes FALL in S2 Turn         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              BONUS: MISMATCH REDUCTION                          │
│  ActuatorNet V1→V2→V3 · Residual Learning · RMA Policy          │
└─────────────────────────────────────────────────────────────────┘
```
---

### Key Technical Challenges Solved

1. **Quaternion Convention** — Isaac Gym (xyzw) vs MuJoCo (wxyz)
2. **Actuator Remapping** — Different joint ordering in ctrl array
3. **Velocity Frame Transformation** — World to body frame
4. **Observation Building** — Reconstruct 48-dim obs vector manually
5. **Torque Clipping Discovery** — Found Isaac Gym's implicit ±30 N·m limit
6. **Velocity-Dependent Dynamics** — Discovered through residual analysis

---

## Author

Disthorn Suttawet | FIBO, KMUTT | January 2026
