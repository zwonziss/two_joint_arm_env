import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
from arm_env import ArmEnv

R = np.deg2rad

ü
PICK_SH  = -20.0;  PICK_EL  = 70.0   
CARRY_SH = -60.0;  CARRY_EL = 20.0   
DROP_SH  = -30.0;  DROP_EL  = 55.0   
STACK_SH = -25.0;  STACK_EL = 68   

PICK_YAW  =   0.0  
DROP_YAW  =  90.0  
PICK2_YAW = -90.0  

G_OPEN = 0.000
G_GRIP = 0.06

BOX_HALF = 0.015


BOX_X  = 0.5;  BOX_Y  =  0.0;  BOX_Z  = BOX_HALF  
BOX2_X = 0.0;   BOX2_Y = -0.5;  BOX2_Z = BOX_HALF  

T = {
    "grip":       (2800, 4800),
    "lift":       (2800, 1200),
    "swing":      (3600, 1600),
    "lower":      (2800, 1600),
    "release":    (1600, 2000),
    "retreat":    (2000, 1200),
    "swing_back": (3600, 1200),
}

def ease(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))

def _set_freejoint(env, joint_name, pos, quat=(1,0,0,0)):
    jnt_id  = env.model.joint(joint_name).id
    adr     = env.model.jnt_qposadr[jnt_id]
    env.data.qpos[adr:adr+3] = pos
    env.data.qpos[adr+3]     = quat[0]
    env.data.qpos[adr+4]     = quat[1]
    env.data.qpos[adr+5]     = quat[2]
    env.data.qpos[adr+6]     = quat[3]
    vel_adr = env.model.jnt_dofadr[jnt_id]
    env.data.qvel[vel_adr:vel_adr+6] = 0.0

def _set_arm_joint(env, joint_name, value):
    jnt_id  = env.model.joint(joint_name).id
    adr     = env.model.jnt_qposadr[jnt_id]
    env.data.qpos[adr] = value
    vel_adr = env.model.jnt_dofadr[jnt_id]
    env.data.qvel[vel_adr] = 0.0

def apply_pose(env, sh_deg, el_deg, grip, yaw_deg=0.0):
    _set_arm_joint(env, "shoulder_yaw", R(yaw_deg))
    _set_arm_joint(env, "shoulder",     R(sh_deg))
    _set_arm_joint(env, "elbow",        R(el_deg))
    _set_arm_joint(env, "fingerL",      grip)
    _set_arm_joint(env, "fingerR",      grip)
    env.data.ctrl[0] = R(sh_deg)
    env.data.ctrl[1] = R(el_deg)
    env.data.ctrl[2] = grip
    env.data.ctrl[3] = R(yaw_deg)
    mujoco.mj_forward(env.model, env.data)

def spawn_boxes(env):
    _set_freejoint(env, "box_joint",  [BOX_X,  BOX_Y,  BOX_Z])
    _set_freejoint(env, "box2_joint", [BOX2_X, BOX2_Y, BOX2_Z])
    mujoco.mj_forward(env.model, env.data)
    print(f"  Red  box at {env.data.body('box').xpos.round(3)}")
    print(f"  Blue box at {env.data.body('box2').xpos.round(3)}")

def P(sh, el, grip, yaw=0.0):
    return np.array([R(sh), R(el), grip, R(yaw)], dtype=np.float64)


def build_seq():
    return [
        
        ("grip",       P(PICK_SH,  PICK_EL,  G_GRIP, PICK_YAW)),   
        ("lift",       P(CARRY_SH, CARRY_EL, G_GRIP, PICK_YAW)),   
        ("swing",      P(CARRY_SH, CARRY_EL, G_GRIP, DROP_YAW)),   
        ("lower",      P(DROP_SH,  DROP_EL,  G_GRIP, DROP_YAW)),   
        ("release",    P(DROP_SH,  DROP_EL,  G_OPEN, DROP_YAW)),   
        ("retreat",    P(CARRY_SH, CARRY_EL, G_OPEN, DROP_YAW)),   

        
        ("swing_back", P(CARRY_SH, CARRY_EL, G_OPEN, PICK2_YAW)),  
        ("lower",      P(PICK_SH,  PICK_EL,  G_OPEN, PICK2_YAW)), 
        ("grip",       P(PICK_SH,  PICK_EL,  G_GRIP, PICK2_YAW)), 
        ("lift",       P(CARRY_SH, CARRY_EL, G_GRIP, PICK2_YAW)), 
        ("swing",      P(CARRY_SH, CARRY_EL, G_GRIP, DROP_YAW)),   
        ("lower",      P(STACK_SH, STACK_EL, G_GRIP, DROP_YAW)),   
        ("release",    P(STACK_SH, STACK_EL, G_OPEN, DROP_YAW)),   
        ("retreat",    P(CARRY_SH, CARRY_EL, G_OPEN, DROP_YAW)),   
    ]

class SM:
    def __init__(self, seq, start):
        self.seq   = seq
        self.i     = 0
        self.ts    = 0
        self.ds    = 0
        self.in_tr = True
        self.cur   = start.copy()

    def tick(self):
        name, tgt = self.seq[self.i]
        tr_n, dw_n = T[name]

        if self.in_tr:
            alpha = ease(self.ts / max(tr_n - 1, 1))
            ctrl  = self.cur + (tgt - self.cur) * alpha
            self.ts += 1
            if self.ts >= tr_n:
                self.cur   = tgt.copy()
                self.in_tr = False
                self.ds    = 0
                print(f"    [{name}] done  ({self.i+1}/{len(self.seq)})")
        else:
            ctrl = self.cur.copy()
            self.ds += 1
            if self.ds >= dw_n:
                self.i += 1
                if self.i >= len(self.seq):
                    return ctrl, True
                self.in_tr = True
                self.ts    = 0
                if self.i == 6:
                    print("\n=== Cycle 2: pick blue → stack on red ===")
                else:
                    print(f"    → {self.seq[self.i][0]}")

        return ctrl, False
def main():
    env = ArmEnv("arm_env.xml")
    env.reset()

    fps      = 60
    video_fps = 120
    duration = 160
    spawn_boxes(env)
    apply_pose(env, 15, 60, G_OPEN, 15)
    
    apply_pose(env, PICK_SH, PICK_EL, G_OPEN, PICK_YAW)

    start = P(PICK_SH, PICK_EL, G_OPEN, PICK_YAW)
    sm    = SM(build_seq(), start)

    frames = []
    last_t = 0.0
    done   = False

    
    render_cam = mujoco.MjvCamera()
    render_cam.distance  = 1.8
    render_cam.azimuth   = 110
    render_cam.elevation = -25
    render_cam.lookat[:] = [0.2, 0.2, 0.1]

    print("\n=== Cycle 1: pick red → place at drop zone ===")

    with mujoco.Renderer(env.model, width=1280, height=720) as renderer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            
            viewer.cam.distance  = 1.5
            viewer.cam.azimuth   = 120
            viewer.cam.elevation = -30
            viewer.cam.lookat[:] = [0.2, 0.2, 0.1]

            while viewer.is_running() and env.data.time < duration and not done:

                ctrl, done = sm.tick()
                env.data.ctrl[0] = ctrl[0]
                env.data.ctrl[1] = ctrl[1]
                env.data.ctrl[2] = ctrl[2]
                env.data.ctrl[3] = ctrl[3]
                env.step(ctrl)

                if env.data.time - last_t >= 1.0 / fps:
                    renderer.update_scene(env.data, camera=render_cam) 
                    frames.append(renderer.render())
                    last_t = env.data.time

                viewer.sync()

            if done:
                for _ in range(400):
                    env.step(ctrl)
                    if env.data.time - last_t >= 1.0 / fps:
                        renderer.update_scene(env.data, camera=render_cam)  
                        frames.append(renderer.render())
                        last_t = env.data.time
                    viewer.sync()
                    if not viewer.is_running():
                        break

                print(f"\n=== Done! ===")
                print(f"  Red  box: {env.data.body('box').xpos.round(3)}")
                print(f"  Blue box: {env.data.body('box2').xpos.round(3)}")

    out = "arm_pick_stack.mp4"
    media.write_video(out, frames, fps=fps*4)
    print(f"Saved {len(frames)} frames ({len(frames)/fps:.1f}s) → {out}")


if __name__ == "__main__":
    main()