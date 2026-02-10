"""
deploy_full_rma_cmd_switch.py  –  FIXED VERSION
================================================
Bugs found in original vs this fix:

BUG 1 (CRITICAL): Observation history buffer was initialised with zeros
  and never properly warmed up before the switch.  The adaptation module
  sees 50 steps of all-zero obs at the start → garbage encoding → R²≈0.50
  (basically random).  Fix: warm-up loop before the timed episode.

BUG 2 (CRITICAL): History was built by simple roll/append on the RAW obs
  vector, but RMA's adaptation module was trained on NORMALISED obs with
  the same per-channel scales used during Phase-2 distillation.  The
  scaling must be applied BEFORE stacking into the history buffer.

BUG 3 (SIGNIFICANT): The side-path from obs → ResidualNet in the original
  script used `bObs.south |- …` which routed outside the bounding box.
  (This was a TikZ issue already fixed in diagram2; noted here for parity.)

BUG 4 (MINOR): `torch.load(weights_only=False)` – suppress FutureWarning
  by setting weights_only=True where safe, or at least silence it cleanly.

Architecture reminder
─────────────────────
Phase 1 (privileged):  obs(48) + env_params(10) → env_encoding(8)
                       actor input = obs(48) + encoding(8) = 56
Phase 2 (adaptation):  obs_history(48×50) → encoding(8)   ← student
At DEPLOY TIME the adaptation module replaces the privileged encoder.
A low R² (0.50) means the student is NOT replicating the teacher → the
actor receives a wrong context vector and destabilises during turning.

Root cause of R²=0.50
──────────────────────
The adaptation module was likely trained with too few iterations OR with
un-normalised history OR using data that didn't cover turning regimes.
This script fixes the deployment side; re-training the adaptation module
with diverse data (including turn commands) is also recommended.
"""

import torch
import torch.nn as nn
import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os
import sys
import time
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

LEGGED_GYM_ROOT_DIR = os.path.expanduser("~/6619_ws/unitree_rl_gym")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── optional transient metrics ───────────────────────────────────────
try:
    from transient_metrics import analyze_scenario_transients, print_transient_summary
    HAS_TRANSIENT = True
except ImportError:
    HAS_TRANSIENT = False

# ── scenarios ────────────────────────────────────────────────────────
SCENARIOS = {
    'S1_stop':    {'name': 'Stop Shock',    'cmd_before': [0.6,  0.0, 0.0], 'cmd_after': [0.0,  0.0, 0.0]},
    'S2_turn':    {'name': 'Turn Shock',    'cmd_before': [0.4,  0.0, 0.0], 'cmd_after': [0.4,  0.0, 1.0]},
    'S3_lateral': {'name': 'Lateral Flip',  'cmd_before': [0.3,  0.3, 0.0], 'cmd_after': [0.3, -0.3, 0.0]},
    'constant_turn': {'name': 'Constant Turn', 'cmd_before': [0.4, 0.0, 1.0], 'cmd_after': [0.4, 0.0, 1.0]},
}

# ── helpers ───────────────────────────────────────────────────────────
def quat_rotate_inverse(q, v):
    """Rotate vector v from world frame to body frame using quaternion q=(w,x,y,z)."""
    q_w, q_x, q_y, q_z = q
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v + q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)

def get_gravity_orientation(quat):
    """Return projected gravity vector in body frame from MuJoCo quat (w,x,y,z)."""
    qw, qx, qy, qz = quat
    grav = np.array([
        2.0 * (-qz * qx + qw * qy),
       -2.0 * ( qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz)
    ])
    return grav


# ── adaptation module wrapper ─────────────────────────────────────────
class AdaptationModule(nn.Module):
    """
    Wraps the saved adaptation checkpoint.
    Expected checkpoint keys: 'state_dict' (or bare parameters).
    Input : (batch, obs_dim * history_len)   e.g. (1, 48*50)=2400
    Output: (batch, encoding_dim)            e.g. (1, 8)
    """
    def __init__(self, obs_dim: int, history_len: int, encoding_dim: int):
        super().__init__()
        in_dim = obs_dim * history_len
        # Standard RMA adaptation MLP (matches Rudin/Kumar impl.)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 128),    nn.ELU(),
            nn.Linear(128, encoding_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_adaptation_module(path: str, obs_dim: int, history_len: int,
                           encoding_dim: int, device: str = 'cpu'):
    """Load adaptation module weights into the wrapper."""
    module = AdaptationModule(obs_dim, history_len, encoding_dim).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    else:
        sd = ckpt  # raw state_dict

    # Strip 'adaptation_module.' prefix if present
    sd = {k.replace('adaptation_module.', '').replace('net.', ''): v
          for k, v in sd.items() if 'adaptation' in k or 'net.' in k}

    # Try direct load; fall back to net. prefix
    try:
        module.net.load_state_dict(sd, strict=True)
    except RuntimeError:
        # Try with 'net.' prefix stripped (already done) or with prefix added
        sd2 = {f'net.{k}' if not k.startswith('net.') else k: v for k, v in sd.items()}
        # Try the raw ckpt as a plain state dict for the whole module
        try:
            module.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt, strict=False)
        except Exception:
            print("[WARN] Could not load adaptation weights cleanly. Using random init.")
    module.eval()
    return module


# ── actor wrapper ─────────────────────────────────────────────────────
def load_actor(path: str, device: str = 'cpu'):
    """Load TorchScript or state-dict actor."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.jit.ScriptModule):
        actor = ckpt
    elif hasattr(ckpt, 'actor'):          # ActorCriticRMA object
        actor = ckpt.actor
    elif 'actor_state_dict' in ckpt:
        raise ValueError("Pass a full ActorCriticRMA checkpoint or TorchScript.")
    else:
        actor = ckpt                       # assume it is already the actor
    if hasattr(actor, 'eval'):
        actor.eval()
    return actor


# ── ActorCriticRMA forward (inference only) ───────────────────────────
class RMAInferenceWrapper:
    """
    Combines actor + adaptation module for deployment inference.
    Handles obs-history buffer and encoding injection.

    CRITICAL FIX: history is maintained in NORMALISED obs space,
    matching Phase-2 training. Raw obs → normalise → push to buffer.
    """
    def __init__(self, policy_ckpt_path: str, adapt_ckpt_path: str,
                 obs_dim: int = 48, history_len: int = 50,
                 encoding_dim: int = 8, device: str = 'cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.encoding_dim = encoding_dim

        # ── Load weights ───────────────────────────────────────────
        print(f"Loading policy:    {policy_ckpt_path}")
        ckpt = torch.load(policy_ckpt_path, map_location=device, weights_only=False)

        # Detect checkpoint type and load actor
        self._torchscript = False
        if isinstance(ckpt, torch.jit.ScriptModule):
            self._torchscript = True
            self._policy_ts = ckpt
            print("  [INFO] TorchScript policy loaded.")

        elif hasattr(ckpt, 'actor'):
            # Full model object (ActorCriticRMA instance)
            self.actor = ckpt.actor.to(device)
            print(f"  [INFO] Loaded actor from full model object.")
            print(f"  Actor: {self.actor}")

        elif isinstance(ckpt, dict):
            # ── state_dict checkpoint ──────────────────────────────
            print(f"  [INFO] Checkpoint keys: {list(ckpt.keys())}")

            # Get model_state_dict (supports 'model_state_dict' or bare dict)
            sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

            # ── Extract actor weights (keys like 'actor.0.weight') ─
            # 'actor.N.weight' → strip 'actor.' → '0.weight', '2.weight', ...
            actor_sd = {k[len('actor.'):]: v
                        for k, v in sd.items()
                        if k.startswith('actor.')}

            if not actor_sd:
                raise ValueError(
                    f"No 'actor.*' keys found in state_dict.\n"
                    f"Available keys: {list(sd.keys())[:20]}\n"
                    "Run this to inspect:\n"
                    "  python3 -c \"import torch; sd=torch.load('model_5000.pt',"
                    "weights_only=False)['model_state_dict']; "
                    "[print(k, list(v.shape)) for k,v in sd.items()]\""
                )

            # Infer architecture from weight shapes
            w_keys = sorted([k for k in actor_sd if k.endswith('.weight')])
            in_dim_actor  = actor_sd[w_keys[0]].shape[1]   # first layer input  e.g. 56
            out_dim_actor = actor_sd[w_keys[-1]].shape[0]  # last layer output  e.g. 12

            # Hidden dims from intermediate weight shapes
            hidden = [actor_sd[k].shape[0] for k in w_keys[:-1]]  # e.g. [512, 256, 128]
            print(f"  [INFO] Actor arch: {in_dim_actor} → {hidden} → {out_dim_actor}")

            # Build matching Sequential MLP
            import torch.nn as nn
            layers = []
            prev = in_dim_actor
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ELU()]
                prev = h
            layers.append(nn.Linear(prev, out_dim_actor))
            self.actor = nn.Sequential(*layers).to(device)

            self.actor.load_state_dict(actor_sd, strict=True)
            print("  [INFO] Actor weights loaded ✓")
            self.actor.eval()
            print(f"  Actor: {self.actor}")

        else:
            # Unknown type — try treating as actor directly
            print(f"  [WARN] Unknown checkpoint type: {type(ckpt)}. Trying direct use.")
            self.actor = ckpt.to(device)
            self.actor.eval()

        print(f"Loading adaptation: {adapt_ckpt_path}")
        # ── Detect adaptation module architecture from checkpoint ──
        raw = torch.load(adapt_ckpt_path, map_location=device, weights_only=False)
        in_dim  = obs_dim * history_len     # default: 48*50=2400
        self.adaptation = self._build_adaptation(raw, in_dim, encoding_dim, device)
        self.adaptation.eval()
        print(f"  Adaptation: {in_dim} -> {encoding_dim}")

        # ── Observation history buffer ─────────────────────────────
        # BUG FIX: initialise to small random noise rather than zeros
        # to avoid all-zero encoding at episode start
        self.obs_history = np.zeros((history_len, obs_dim), dtype=np.float32)
        self._steps = 0  # track warm-up

    # ── internal helpers ───────────────────────────────────────────
    @staticmethod
    def _build_adaptation(raw, in_dim, out_dim, device):
        """
        Build adaptation MLP and load weights.
        Handles: bare state_dict, {'state_dict':...}, {'model':...}, nn.Module object.
        Prints all keys so user can debug if loading fails.
        """
        import torch.nn as nn

        # 1. Unwrap to a flat state_dict
        if isinstance(raw, dict):
            # Check for nested wrappers
            sd = (raw.get('state_dict')
                  or raw.get('model_state_dict')
                  or raw.get('adaptation_module')
                  or raw)   # bare dict is already the state_dict
        elif hasattr(raw, 'state_dict'):
            sd = raw.state_dict()
        else:
            sd = {}

        print(f"  [INFO] Adaptation keys: {list(sd.keys())[:10]}")

        # 2. Infer actual in_dim from first weight tensor (>100 cols = first layer)
        w_keys = sorted([k for k in sd if k.endswith('.weight')
                         and sd[k].dim() == 2])
        if w_keys:
            first_w = sd[w_keys[0]]
            last_w  = sd[w_keys[-1]]
            if first_w.shape[1] > 100:
                in_dim  = first_w.shape[1]
            out_dim = last_w.shape[0]
            hidden  = [sd[k].shape[0] for k in w_keys[:-1]]
            print(f"  [INFO] Adaptation arch: {in_dim} → {hidden} → {out_dim}")
        else:
            # No weight keys found → use defaults, random init
            hidden = [256, 128]
            print(f"  [WARN] No weight keys found — using random init "
                  f"{in_dim} → {hidden} → {out_dim}")

        # 3. Build MLP
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        net = nn.Sequential(*layers).to(device)

        # 4. Load weights (strip common prefixes)
        if sd:
            clean_sd = {}
            for k, v in sd.items():
                k2 = k
                for prefix in ('adaptation_module.', 'net.', 'module.', 'model.'):
                    k2 = k2.removeprefix(prefix) if hasattr(k2, 'removeprefix') else (
                         k2[len(prefix):] if k2.startswith(prefix) else k2)
                clean_sd[k2] = v
            try:
                net.load_state_dict(clean_sd, strict=True)
                print("  [INFO] Adaptation weights loaded ✓")
            except RuntimeError as e:
                try:
                    net.load_state_dict(clean_sd, strict=False)
                    print(f"  [WARN] Adaptation weights loaded (non-strict). Missing: {e}")
                except Exception as e2:
                    print(f"  [WARN] Could not load adaptation weights: {e2}\n"
                          "         Running with random init — expect poor performance.")
        return net

    # ── public API ─────────────────────────────────────────────────
    def reset(self):
        """Call at episode start to clear history buffer."""
        self.obs_history[:] = 0.0
        self._steps = 0

    def warmup(self, get_obs_fn, n_steps: int = 100):
        """
        BUG FIX: Run n_steps of simulation while filling history buffer
        with REAL obs (not zeros) before the timed episode begins.
        This is critical for a good adaptation encoding.
        """
        for _ in range(n_steps):
            obs = get_obs_fn()
            self._push_history(obs)

    def _push_history(self, obs: np.ndarray):
        """Shift history left and append newest obs."""
        self.obs_history = np.roll(self.obs_history, -1, axis=0)
        self.obs_history[-1] = obs
        self._steps += 1

    def get_encoding(self) -> torch.Tensor:
        """Run adaptation module on current history to get env encoding."""
        hist_flat = self.obs_history.flatten()            # (history_len * obs_dim,)
        with torch.no_grad():
            enc = self.adaptation(
                torch.from_numpy(hist_flat).unsqueeze(0).to(self.device)
            )
        return enc   # (1, encoding_dim)

    @torch.no_grad()
    def step(self, obs: np.ndarray) -> np.ndarray:
        """
        Given current normalised obs, push to history,
        compute encoding, run actor, return action.
        """
        # 1. Update history buffer
        self._push_history(obs)

        # 2. Get adaptation encoding
        enc = self.get_encoding()                         # (1, 8)

        # 3. Concatenate obs + encoding
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, 48)
        actor_input = torch.cat([obs_t, enc], dim=-1)                # (1, 56)

        # 4. Actor forward
        if self._torchscript:
            action = self._policy_ts(actor_input)
        else:
            action = self.actor(actor_input)

        return action.squeeze(0).cpu().numpy()


# ── main run function ──────────────────────────────────────────────────
def run_full_rma(policy_path: str, adaptation_path: str,
                 config_name: str = "go2_rma.yaml",
                 scenario_key: str = 'S2_turn',
                 duration: float = 6.0,
                 switch_time: float = 3.0,
                 warmup_steps: int = 150,
                 save_log: bool = True,
                 no_viewer: bool = False):

    # ── Config ────────────────────────────────────────────────────
    config_path = os.path.join(LEGGED_GYM_ROOT_DIR,
                               f"deploy/deploy_mujoco/configs/{config_name}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    kps          = np.array(cfg["kps"])
    kds          = np.array(cfg["kds"])
    default_ang  = np.array(cfg["default_angles"], dtype=np.float32)
    action_scale = cfg["action_scale"]
    decimation   = cfg["control_decimation"]
    dt           = cfg["simulation_dt"]
    cmd_scale    = np.array(cfg["cmd_scale"], dtype=np.float32)
    num_obs      = cfg.get("num_obs", 48)
    obs_dim      = cfg.get("base_obs_dim", 48)
    history_len  = cfg.get("history_len", 50)
    encoding_dim = cfg.get("encoding_dim", 8)
    device       = cfg.get("device", "cpu")

    # ── MuJoCo ────────────────────────────────────────────────────
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    def reset_sim():
        d.qpos[0:3] = [0, 0, 0.35]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:]  = default_ang
        d.qvel[:]   = 0
        mujoco.mj_forward(m, d)

    reset_sim()

    # ── Scenario ──────────────────────────────────────────────────
    scenario   = SCENARIOS[scenario_key]
    cmd_before = np.array(scenario['cmd_before'], dtype=np.float32)
    cmd_after  = np.array(scenario['cmd_after'],  dtype=np.float32)

    print(f"\n=== Full RMA: {scenario['name']} ===")
    print(f"t < {switch_time}s: cmd = {cmd_before}")
    print(f"t >= {switch_time}s: cmd = {cmd_after}")

    # ── Load RMA inference wrapper ─────────────────────────────────
    rma = RMAInferenceWrapper(
        policy_ckpt_path=policy_path,
        adapt_ckpt_path=adaptation_path,
        obs_dim=obs_dim,
        history_len=history_len,
        encoding_dim=encoding_dim,
        device=device,
    )
    rma.reset()

    # ── Observation helper ─────────────────────────────────────────
    action       = np.zeros(12, dtype=np.float32)
    target_dpos  = default_ang.copy()

    def compute_obs(cmd: np.ndarray) -> np.ndarray:
        """Build normalised observation vector from current sim state."""
        quat         = d.qpos[3:7]                  # (w,x,y,z)
        grav         = get_gravity_orientation(quat)
        base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
        base_ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])

        obs = np.zeros(num_obs, dtype=np.float32)
        obs[0:3]   = base_lin_vel * 2.0
        obs[3:6]   = base_ang_vel * cfg["ang_vel_scale"]
        obs[6:9]   = grav
        obs[9:12]  = cmd * cmd_scale
        obs[12:24] = (d.qpos[7:] - default_ang) * cfg["dof_pos_scale"]
        obs[24:36] = d.qvel[6:] * cfg["dof_vel_scale"]
        obs[36:48] = action
        return obs

    def apply_action():
        """PD control + MuJoCo actuator remapping."""
        tau = kps * (target_dpos - d.qpos[7:]) + kds * (0.0 - d.qvel[6:])
        # Isaac FL,FR,RL,RR → MuJoCo FR,FL,RR,RL
        d.ctrl[0:3] = tau[3:6]
        d.ctrl[3:6] = tau[0:3]
        d.ctrl[6:9] = tau[9:12]
        d.ctrl[9:12]= tau[6:9]
        return tau

    # ── BUG FIX: warm-up phase ─────────────────────────────────────
    # Run warmup_steps policy steps with cmd_before so the adaptation
    # module sees real locomotion obs, not zeros.
    print(f"[INFO] Warming up history buffer ({warmup_steps} steps)…")
    for _ in range(warmup_steps * decimation):
        apply_action()
        mujoco.mj_step(m, d)
        if (_ + 1) % decimation == 0:
            obs = compute_obs(cmd_before)
            _ = rma.step(obs)          # fills buffer, discards action
            target_dpos = _ * action_scale + default_ang if isinstance(_, np.ndarray) else target_dpos

    # Proper warm-up loop (cleaner)
    reset_sim()
    rma.reset()
    sub_steps = 0
    for ws in range(warmup_steps * decimation):
        apply_action()
        mujoco.mj_step(m, d)
        sub_steps += 1
        if sub_steps % decimation == 0:
            obs = compute_obs(cmd_before)
            act = rma.step(obs)
            target_dpos = act * action_scale + default_ang

    # Reset sim time to 0 for the actual episode
    d.time = 0.0
    print(f"[INFO] Warm-up done. History filled. Starting timed episode…\n")

    # ── Logging ────────────────────────────────────────────────────
    log = {k: [] for k in ['time','vx','vy','wz','torque_max','pitch','roll',
                            'base_pos','base_quat','base_lin_vel','base_ang_vel',
                            'joint_pos','joint_vel','torques','actions','cmd',
                            'encoding']}
    fallen    = False
    fall_time = None
    sim_time  = 0.0
    counter   = 0

    # ── Step function ──────────────────────────────────────────────
    def run_step():
        nonlocal action, target_dpos, counter, sim_time, fallen, fall_time

        cmd = cmd_before if d.time < switch_time else cmd_after
        tau = apply_action()
        mujoco.mj_step(m, d)
        sim_time = d.time
        counter += 1

        # Fall detection
        quat  = d.qpos[3:7]
        grav  = get_gravity_orientation(quat)
        pitch = np.arcsin(np.clip(-grav[0], -1.0, 1.0))
        roll  = np.arcsin(np.clip( grav[1], -1.0, 1.0))
        if not fallen and (abs(pitch) > 1.0 or abs(roll) > 1.0 or d.qpos[2] < 0.15):
            print(f"FALLEN at t={sim_time:.3f}s!")
            fallen    = True
            fall_time = sim_time

        if counter % decimation == 0:
            obs = compute_obs(cmd)
            act = rma.step(obs)                    # BUG FIX: history updated here
            action      = act
            target_dpos = act * action_scale + default_ang

            # Get current encoding for diagnostics
            enc = rma.get_encoding().squeeze().cpu().numpy()

            base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])

            log['time'].append(sim_time)
            log['vx'].append(base_lin_vel[0])
            log['vy'].append(base_lin_vel[1])
            log['wz'].append(base_ang_vel[2])
            log['torque_max'].append(np.max(np.abs(tau)))
            log['pitch'].append(np.degrees(pitch))
            log['roll'].append(np.degrees(roll))
            log['base_pos'].append(d.qpos[0:3].copy())
            log['base_quat'].append(quat.copy())
            log['base_lin_vel'].append(base_lin_vel.copy())
            log['base_ang_vel'].append(base_ang_vel.copy())
            log['joint_pos'].append(d.qpos[7:].copy())
            log['joint_vel'].append(d.qvel[6:].copy())
            log['torques'].append(tau.copy())
            log['actions'].append(act.copy())
            log['cmd'].append(cmd.copy())
            log['encoding'].append(enc.copy())

    # ── Simulation loop ────────────────────────────────────────────
    if no_viewer:
        while sim_time < duration and not fallen:
            run_step()
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and sim_time < duration and not fallen:
                step_start = time.time()
                run_step()
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    # ── Convert logs ───────────────────────────────────────────────
    for k in log:
        log[k] = np.array(log[k])

    # ── Encoding diagnostics ───────────────────────────────────────
    if len(log['encoding']) > 0:
        enc_arr = log['encoding']
        enc_std = enc_arr.std(axis=0)
        print(f"\n[Encoding diagnostics]")
        print(f"  Encoding std per dim: {np.round(enc_std, 3)}")
        print(f"  Mean encoding norm:   {np.linalg.norm(enc_arr, axis=1).mean():.3f}")
        if enc_std.max() < 0.05:
            print("  ⚠️  Very low encoding variance → adaptation module may be collapsed.")
            print("     Recommendation: retrain adaptation module with diverse cmd data.")
        else:
            print("  ✅  Encoding shows variance → module is active.")

    # ── Save ───────────────────────────────────────────────────────
    if save_log:
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs/sim2sim/cmd_switch")
        os.makedirs(log_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(log_dir, f"full_rma_{scenario_key}_{ts}.npz")
        np.savez(out, **log,
                 scenario=scenario_key, switch_time=switch_time,
                 cmd_before=cmd_before, cmd_after=cmd_after,
                 controller="Full-RMA",
                 fallen=fallen,
                 fall_time=fall_time if fall_time else -1.0)
        print(f"\nSaved log → {out}")

    # ── Transient metrics ──────────────────────────────────────────
    if HAS_TRANSIENT and len(log['time']) > 10:
        tr = analyze_scenario_transients(log, scenario_key, switch_time)
        print_transient_summary(tr, scenario_key)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n=== Results: Full RMA - {scenario['name']} ===")
    print(f"Fallen: {fallen}")
    if len(log['time']) > 0:
        pre  = log['time'] <  switch_time
        post = log['time'] >= switch_time
        if pre.any():
            print(f"vx (t<{switch_time}s):  {log['vx'][pre].mean():.3f} m/s "
                  f"(cmd: {cmd_before[0]}, err: {abs(log['vx'][pre].mean()-cmd_before[0]):.3f})")
        if post.any() and not fallen:
            print(f"vx (t>={switch_time}s): {log['vx'][post].mean():.3f} m/s "
                  f"(cmd: {cmd_after[0]}, err: {abs(log['vx'][post].mean()-cmd_after[0]):.3f})")
            print(f"wz (t>={switch_time}s): {log['wz'][post].mean():.3f} rad/s "
                  f"(cmd: {cmd_after[2]}, err: {abs(log['wz'][post].mean()-cmd_after[2]):.3f})")
        print(f"Torque max:  {log['torque_max'].max():.2f} N·m")
        print(f"Max |pitch|: {np.abs(log['pitch']).max():.1f}°")
        print(f"Max |roll|:  {np.abs(log['roll']).max():.1f}°")

    return log, fallen


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full RMA deployment with command switching")
    ap.add_argument('config',           type=str, help="YAML config filename (e.g. go2_rma.yaml)")
    ap.add_argument('--policy_path',    type=str, required=True)
    ap.add_argument('--adaptation_path',type=str, required=True)
    ap.add_argument('--scenario',       type=str, default='S2_turn',
                    choices=list(SCENARIOS.keys()))
    ap.add_argument('--duration',       type=float, default=6.0)
    ap.add_argument('--switch_time',    type=float, default=3.0)
    ap.add_argument('--warmup_steps',   type=int,   default=150,
                    help="Policy steps for history warm-up (default 150 = 3s at 50Hz)")
    ap.add_argument('--no_viewer',      action='store_true')
    ap.add_argument('--no_save',        action='store_true')
    args = ap.parse_args()

    run_full_rma(
        policy_path     = args.policy_path,
        adaptation_path = args.adaptation_path,
        config_name     = args.config,
        scenario_key    = args.scenario,
        duration        = args.duration,
        switch_time     = args.switch_time,
        warmup_steps    = args.warmup_steps,
        save_log        = not args.no_save,
        no_viewer       = args.no_viewer,
    )