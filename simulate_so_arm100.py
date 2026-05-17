import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
TCP_SITE = "ee_site"
BOX_BODY = "box"
ROT_BODY = "Rotation_Pitch"
ROT_LIMITS = (-1.92, 1.92)
YAW_OFFSET = np.pi / 2

def get_cube_half_height():
    bid = model.body(BOX_BODY).id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid:
            return float(max(model.geom_size[g][2], model.geom_size[g][0]))
    return 0.013

def solve_ik(target_pos, approach=np.array([0., 0., -1.]), seed=None, 
             max_iters=1000, pos_tol=1e-3, ori_tol=0.1, lock_joints=(), ori_weight=0.05):
    
    site_id = model.site(TCP_SITE).id
    free_joints = [n for n in ARM_JOINTS if n not in lock_joints]
    
    dof_ids = [model.joint(n).dofadr[0] for n in free_joints]
    qpos_ids = [model.joint(n).qposadr[0] for n in free_joints]
    ranges = [model.jnt_range[model.joint(n).id] for n in free_joints]

    if seed:
        for n, v in seed.items():
            data.qpos[model.joint(n).qposadr[0]] = v

    approach = approach / np.linalg.norm(approach)
    I3, I6 = np.eye(3), np.eye(6)

    for _ in range(1000):
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.site(site_id).xpos
        if np.linalg.norm(pos_err) < pos_tol:
            break
            
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, dof_ids]
        
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * I3, pos_err)
        for i, qa in enumerate(qpos_ids):
            data.qpos[qa] = np.clip(data.qpos[qa] + 0.5 * dq[i], ranges[i][0], ranges[i][1])

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_pos = data.site(site_id).xpos
        tool_axis = data.site(site_id).xmat.reshape(3, 3)[:, 2]
        
        pos_err = target_pos - cur_pos
        ori_err = np.cross(tool_axis, approach)
        
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
            break
            
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        
        err = np.concatenate([pos_err, ori_weight * ori_err])
        dq = J.T @ np.linalg.solve(J @ J.T + (0.08 ** 2) * I6, err)
        
        for i, qa in enumerate(qpos_ids):
            data.qpos[qa] = np.clip(data.qpos[qa] + 0.4 * dq[i], ranges[i][0], ranges[i][1])

    mujoco.mj_forward(model, data)
    return {n: float(data.qpos[model.joint(n).qposadr[0]]) for n in ARM_JOINTS}, np.linalg.norm(target_pos - data.site(site_id).xpos)

def solve_ik_multi(target_pos, approach, seeds, max_iters=1000):
    best_q, best_err = None, float("inf")
    qpos_save = data.qpos.copy()
    
    for s in seeds:
        data.qpos[:] = qpos_save
        q, err = solve_ik(target_pos, approach, seed=s, max_iters=max_iters)
        if err < best_err:
            best_q, best_err = q, err
            
    for n, v in best_q.items():
        data.qpos[model.joint(n).qposadr[0]] = v
    mujoco.mj_forward(model, data)
    
    return best_q

def set_ctrl(target):
    for name, val in target.items():
        data.ctrl[model.actuator(name).id] = val

def interpolate(start, end, steps, viewer, hold=0):
    keys = list(end.keys())
    for i in range(steps):
        alpha = i / max(steps - 1, 1)
        ctrl = {k: start.get(k, data.ctrl[model.actuator(k).id]) * (1 - alpha) + end[k] * alpha for k in keys}
        set_ctrl(ctrl)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        
    for _ in range(hold):
        set_ctrl(end)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

def plan_pick_and_place(place_xy, z_standoff=0.25, lift_height=0.3, wrist_roll=1.5, jaw_open=0.5, jaw_close=-0.174):
    mujoco.mj_forward(model, data)
    box_pos = data.body(BOX_BODY).xpos.copy()
    pivot_xy = data.body(ROT_BODY).xpos[:2].copy()
    cube_h = get_cube_half_height()
    qpos_backup = data.qpos.copy()

    get_yaw = lambda xy: float(np.arctan2(xy[1] - pivot_xy[1], xy[0] - pivot_xy[0]))
    pick_yaw, place_yaw = get_yaw(box_pos[:2]), get_yaw(place_xy)

    offset = -0.012 if box_pos[0] < 0 else -0.015
    offset_2 = 0.02 if box_pos[0] < 0 else -0.015
    
    grasp_center = np.array([box_pos[0] + (offset * 0.5), box_pos[1] + offset_2, cube_h + 0.002])
    place_center = np.array([place_xy[0], place_xy[1], cube_h])
    DOWN = np.array([0, 0, -1])

    targets = {
        "pre": grasp_center + [0, 0, z_standoff],
        "grab": grasp_center.copy(),
        "lift": grasp_center + [0, 0, lift_height],
        "over_pl": place_center + [0, 0, lift_height],
        "pre_pl": place_center + [0, 0, z_standoff],
        "drop": place_center.copy()
    }

    def get_seeds(yaw):
        rot = np.clip(yaw + YAW_OFFSET, ROT_LIMITS[0], ROT_LIMITS[1])
        return [
            {"Rotation": rot, "Pitch": -1.5, "Elbow": 1.5, "Wrist_Pitch": -0.3, "Wrist_Roll": 0.0},
            {"Rotation": rot, "Pitch": -1.0, "Elbow": 1.8, "Wrist_Pitch": -0.3, "Wrist_Roll": 0.0},
            {"Rotation": rot, "Pitch": -2.0, "Elbow": 1.0, "Wrist_Pitch": -0.3, "Wrist_Roll": 0.0},
            {"Rotation": rot, "Pitch": -2.5, "Elbow": 2.2, "Wrist_Pitch": -0.3, "Wrist_Roll": 0.0},
        ]

    def solve(xyz, seeds):
        q = solve_ik_multi(np.asarray(xyz), DOWN, seeds)
        q["Wrist_Roll"] = wrist_roll
        return q

    pick_seeds = get_seeds(pick_yaw)
    place_seeds = get_seeds(place_yaw)

    q_pre = solve(targets["pre"], pick_seeds)
    q_pick = solve(targets["grab"], [q_pre] + pick_seeds)
    q_pick["Wrist_Pitch"] = -0.85 # manual grip angle tweak
    
    q_lift = solve(targets["lift"], [q_pick] + pick_seeds)
    q_over = solve(targets["over_pl"], [q_lift] + place_seeds)
    q_pre_pl = solve(targets["pre_pl"], [q_over] + place_seeds)
    q_drop = solve(targets["drop"], [q_pre_pl] + place_seeds)

    data.qpos[:] = qpos_backup
    mujoco.mj_forward(model, data)
    
    return [
        {**q_pre, "Jaw": jaw_open},      
        {**q_pick, "Jaw": jaw_open},     
        {**q_pick, "Jaw": jaw_close},    
        {**q_lift, "Jaw": jaw_close},    
        {**q_over, "Jaw": jaw_close},    
        {**q_pre_pl, "Jaw": jaw_close},  
        {**q_drop, "Jaw": jaw_close},    
        {**q_drop, "Jaw": jaw_open},    
        {**q_pre_pl, "Jaw": jaw_open}    
    ]

if __name__ == "__main__":
    home = {"Rotation": 0.0, "Pitch": -2.0, "Elbow": 0.9,
            "Wrist_Pitch": 0.5, "Wrist_Roll": 0.0, "Jaw": 0.5}

    STEPS, HOLD = 300, 150
    PLACE_XY = (-0.2, -0.22)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_ctrl(home)
        for _ in range(500):
            mujoco.mj_step(model, data)
        viewer.sync()

        waypoints = plan_pick_and_place(PLACE_XY)
        
        timings = [
            (STEPS, HOLD),          
            (STEPS, HOLD),           
            (STEPS // 2, HOLD * 4), 
            (STEPS * 2, HOLD * 2),   
            (STEPS * 4, HOLD),      
            (STEPS * 2, HOLD),       
            (STEPS, HOLD),          
            (STEPS // 2, HOLD * 2), 
            (STEPS, HOLD)         
        ]

        curr_state = home
        for next_state, (s, h) in zip(waypoints, timings):
            interpolate(curr_state, next_state, s, viewer, h)
            curr_state = next_state

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)