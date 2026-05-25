import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class FrankaTrackingEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            "mujoco_menagerie/franka_emika_panda/scene.xml"
        )

        self.data = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand"
        )

        self.steps=0

        # RL learns 7 joint torque corrections
        self.action_space=spaces.Box(
            low=-5,
            high=5,
            shape=(7,),
            dtype=np.float32
        )

        # Observation: end-effector position + velocity + tracking error
        self.observation_space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )


    def get_obs(self):

        qvel=self.data.qvel[:7]

        # Add Gaussian noise
        noise_std=0.01

        qvel=qvel+np.random.normal(
            0,
            noise_std,
            7
        )

        t=self.data.time

        radius=0.15

        cpos_d=np.array([
            0.5+radius*np.cos(t),
            radius*np.sin(t),
            0.5
        ])

        cpos=self.data.xpos[self.ee_id]

        J_pos=np.zeros((3,self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            J_pos,
            None,
            self.ee_id
        )

        J=J_pos[:,:7]

        cvel=J@qvel

        c_error=cpos_d-cpos

        return np.concatenate([
            cpos,
            cvel,
            c_error
        ]).astype(np.float32)



    def reset(self,seed=None,options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.steps=0

        return self.get_obs(),{}



    def step(self,action):

        obs=self.get_obs()

        c_error=obs[6:9]
        cvel=obs[3:6]

        t=self.data.time
        radius=0.15

        cvel_d=np.array([
            -radius*np.sin(t),
            radius*np.cos(t),
            0
        ])

        cvel_error=cvel_d-cvel


        J_pos=np.zeros((3,self.model.nv))

        mujoco.mj_jacBody(
            self.model,
            self.data,
            J_pos,
            None,
            self.ee_id
        )

        J=J_pos[:,:7]


        Kp=np.diag([150,150,150])
        Kd=np.diag([30,30,30])

        force=(
            Kp@c_error
            +
            Kd@cvel_error
        )

        tau_PD=J.T@force

        # Combine PD controller with RL correction
        tau=tau_PD+action

        tau=np.clip(
            tau,
            -50,
            50
        )

        self.data.ctrl[:7]=tau
        self.data.ctrl[7]=0

        mujoco.mj_step(
            self.model,
            self.data
        )


        # Reward: minimise tracking error
        reward=(
            -np.linalg.norm(c_error)
            -
            0.001*np.sum(action**2)
        )


        self.steps+=1

        truncated=self.steps>1000

        return (
            self.get_obs(),
            reward,
            False,
            truncated,
            {}
        )