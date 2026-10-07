import json, os, pickle
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(ROOT, "arm.xml")
STATE_PATH = os.path.join(ROOT, "state", "state.pkl")
LOG_PATH = os.path.join(ROOT, "state", "log.jsonl")
OBS_PATH = os.path.join(ROOT, "state", "obs.json")
FRAMES_DIR = os.path.join(ROOT, "frames")

TABLE_TOP_Z = 0.30
FLOOR_Z = 0.0

def load_model():
    m = mujoco.MjModel.from_xml_string(open(XML).read())
    d = mujoco.MjData(m)
    return m, d

def save_state(m, d, target_ee, target_grip, step_idx):
    with open(STATE_PATH, "wb") as f:
        pickle.dump({
            "qpos": d.qpos.copy(),
            "qvel": d.qvel.copy(),
            "ctrl": d.ctrl.copy(),
            "target_ee": target_ee,
            "target_grip": target_grip,
            "step_idx": step_idx,
        }, f)

def load_state(m, d):
    with open(STATE_PATH, "rb") as f:
        s = pickle.load(f)
    d.qpos[:] = s["qpos"]
    d.qvel[:] = s["qvel"]
    d.ctrl[:] = s["ctrl"]
    mujoco.mj_forward(m, d)
    return s["target_ee"], s["target_grip"], s["step_idx"]

def ee_pos(m, d):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    return d.site_xpos[sid].copy()

def cube_pos(m, d, name="cube_a"):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return d.xpos[bid].copy()

def cube_vel(m, d, name="cube_a"):
    jid = m.joint(f"{name}_free").id
    dadr = m.jnt_dofadr[jid]
    return d.qvel[dadr:dadr+3].copy()  # linear velocity of the free joint

def slot_pos(m, d, name="slot_a_site"):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    return d.site_xpos[sid].copy()

ARM_JOINT_NAMES = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch"]
ARM_ACT_NAMES = ["a_base_yaw", "a_shoulder", "a_elbow", "a_wrist"]

def solve_ik(m, d, target_xyz, n_restarts=6, iters=250, damping=0.12, max_dq=0.08):
    """Offline (kinematics-only, no dynamics) damped-least-squares IK for the 4 arm
    joints, with a few random restarts to escape local minima / near-singular poses.
    Returns the best joint-angle target found (does not mutate d)."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    dof_ids = [m.jnt_dofadr[j] for j in joint_ids]
    qadr = [m.jnt_qposadr[j] for j in joint_ids]
    jnt_range = np.array([m.jnt_range[j] for j in joint_ids])

    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos[:]
    cur0 = np.array([d.qpos[a] for a in qadr])

    best_q, best_err = None, np.inf
    rng = np.random.default_rng(0)
    for restart in range(n_restarts):
        if restart == 0:
            q = cur0.copy()
        else:
            q = rng.uniform(jnt_range[:, 0], jnt_range[:, 1])
        for a, val in zip(qadr, q):
            scratch.qpos[a] = val
        mujoco.mj_forward(m, scratch)
        for _ in range(iters):
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, scratch, jacp, None, sid)
            J = jacp[:, dof_ids]
            err = target_xyz - scratch.site_xpos[sid]
            err_norm = np.linalg.norm(err)
            if err_norm < 1e-4:
                break
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -max_dq, max_dq)
            q = q + dq
            q = np.clip(q, jnt_range[:, 0], jnt_range[:, 1])
            for a, val in zip(qadr, q):
                scratch.qpos[a] = val
            mujoco.mj_forward(m, scratch)
        final_err = np.linalg.norm(target_xyz - scratch.site_xpos[sid])
        if final_err < best_err:
            best_err = final_err
            best_q = q.copy()
        if best_err < 1e-3:
            break
    return best_q, best_err

def apply_grip(m, d, grip_val):
    # grip_val in [0,1]; 0 = open (0.045), 1 = closed (0.0)
    finger_target = (1.0 - grip_val) * 0.045
    d.ctrl[m.actuator("a_finger_l").id] = finger_target
    d.ctrl[m.actuator("a_finger_r").id] = finger_target

def gripper_closure(m, d):
    fl = d.qpos[m.jnt_qposadr[m.joint("finger_l_joint").id]]
    fr = d.qpos[m.jnt_qposadr[m.joint("finger_r_joint").id]]
    return 1.0 - ((fl + fr) / 2.0) / 0.045  # 0=open .. 1=closed

def render(m, d, path, width=480, height=360):
    with mujoco.Renderer(m, height=height, width=width) as r:
        r.update_scene(d, camera="front")
        img = r.render()
    from PIL import Image
    Image.fromarray(img).save(path)

CUBE_HALF = 0.022
DISTURBANCE_STEP_THRESH = 0.006   # per-step unintended xy movement worth flagging
DISTURBANCE_ALERT_THRESH = 0.025  # cumulative unintended drift that should stop the loop

def _read_log_lines():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]

def obs_dict(m, d, target_ee, target_grip, step_idx, extra=None):
    ep = ee_pos(m, d)
    sa = slot_pos(m, d, "slot_a_site")
    sb = slot_pos(m, d, "slot_b_site")
    cubes = {name: cube_pos(m, d, name) for name in ("cube_a", "cube_b")}
    vels = {name: cube_vel(m, d, name) for name in ("cube_a", "cube_b")}
    closure = gripper_closure(m, d)

    grasped_name = None
    for name, cp in cubes.items():
        is_grasped = (
            closure > 0.35
            and np.linalg.norm(cp[:2] - ep[:2]) < 0.05
            and abs(cp[2] - ep[2]) < 0.08
            and cp[2] > TABLE_TOP_Z + 0.01
        )
        if is_grasped:
            grasped_name = name

    def settled(name):
        return bool(np.linalg.norm(vels[name]) < 0.05)

    fell = {name: bool(cp[2] < TABLE_TOP_Z - 0.05 and cp[2] > FLOOR_Z - 0.02) for name, cp in cubes.items()}
    on_floor = {name: bool(cp[2] < 0.05) for name, cp in cubes.items()}

    ca, cb = cubes["cube_a"], cubes["cube_b"]
    a_in_slot_a = np.linalg.norm(ca[:2] - sa[:2]) < 0.06 and ca[2] < TABLE_TOP_Z + 0.05
    b_in_slot_b = np.linalg.norm(cb[:2] - sb[:2]) < 0.06 and cb[2] < TABLE_TOP_Z + 0.05
    load_success = bool(
        grasped_name is None and a_in_slot_a and b_in_slot_b
        and settled("cube_a") and settled("cube_b")
    )

    # --- drift tracking: catch "small nudge every step" trends, not just single-step error ---
    prior = _read_log_lines()
    disturbance = {}
    cumulative_drift = {}
    for name, cp in cubes.items():
        was_held_last = bool(prior) and prior[-1].get("grasped") == name
        step_drift = 0.0
        if prior and not was_held_last and grasped_name != name:
            prev_pos = np.array(prior[-1][f"{name}_pos"])
            step_drift = float(np.linalg.norm(cp[:2] - prev_pos[:2]))
        prev_cum = prior[-1].get(f"{name}_cumulative_drift", 0.0) if prior else 0.0
        cum = prev_cum + (step_drift if step_drift > DISTURBANCE_STEP_THRESH else 0.0)
        disturbance[name] = round(step_drift, 4)
        cumulative_drift[name] = round(cum, 4)

    disturbance_alert = any(v > DISTURBANCE_ALERT_THRESH for v in cumulative_drift.values())

    o = {
        "step": step_idx,
        "ee_pos": ep.round(3).tolist(),
        "cube_a_pos": ca.round(3).tolist(),
        "cube_b_pos": cb.round(3).tolist(),
        "slot_a_pos": sa.round(3).tolist(),
        "slot_b_pos": sb.round(3).tolist(),
        "gripper_closure": round(closure, 2),
        "grasped": grasped_name,
        "ee_to_cube_a_dist": round(float(np.linalg.norm(ep - ca)), 3),
        "ee_to_cube_b_dist": round(float(np.linalg.norm(ep - cb)), 3),
        "cube_a_to_slot_a_dist_xy": round(float(np.linalg.norm(ca[:2] - sa[:2])), 3),
        "cube_b_to_slot_b_dist_xy": round(float(np.linalg.norm(cb[:2] - sb[:2])), 3),
        "cube_a_fell_off_table": fell["cube_a"],
        "cube_b_fell_off_table": fell["cube_b"],
        "cube_a_on_floor": on_floor["cube_a"],
        "cube_b_on_floor": on_floor["cube_b"],
        "cube_a_step_drift": disturbance["cube_a"],
        "cube_b_step_drift": disturbance["cube_b"],
        "cube_a_cumulative_drift": cumulative_drift["cube_a"],
        "cube_b_cumulative_drift": cumulative_drift["cube_b"],
        "disturbance_alert": bool(disturbance_alert),
        "task_success": load_success,
        "commanded_target_ee": None if target_ee is None else list(np.round(target_ee, 3)),
        "commanded_grip": target_grip,
    }
    if extra:
        o.update(extra)
    with open(OBS_PATH, "w") as f:
        json.dump(o, f, indent=2)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(o) + "\n")
    return o