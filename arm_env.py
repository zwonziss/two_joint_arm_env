import mujoco
import numpy as np


class ArmEnv:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        ctrl_range = self.model.actuator_ctrlrange
        action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[:] = action

        # CHANGED: 1 physics step per call (was 10)
        # timestep=0.002s, so each step() = 0.002s of simulation
        mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()
        reward = self.compute_reward()
        done = False
        return obs, reward, done

    def get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

    def compute_reward(self):
        box_pos = self.data.body("box").xpos.copy()
        target_pos = np.array([1, 0, 1])
        distance = np.linalg.norm(box_pos - target_pos)
        reward = -distance
        if distance < 0.05:
            reward += 10
        return reward