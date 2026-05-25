import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)
ee_id=mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "hand"

)

# Load RL model
model_rl=PPO.load("franka_policy")

# Logging
actual_positions=[]
desired_positions=[]
tracking_errors=[]
times=[]

with mujoco.viewer.launch_passive(model,data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # Read current arm joint velocities
    # Use only the first 7 values (arm joints) and ignore the gripper
    qvel = data.qvel[:7]

    t = data.time # Simulation time used to generate trajectories

    # Circle trajectory
    radius = 0.15
    x = 0.5 + radius*np.cos(t)
    y = radius*np.sin(t)
    z = 0.5

    # Desired end-effector (cartesian) trajectories
    cpos_d = np.array([x,y,z])

    cvel_d = np.array([
        -radius*np.sin(t),
        radius*np.cos(t),
        0
    ])
    
    # Current end-effector position
    cpos=data.xpos[ee_id]
    
    # Jacobian
    J_pos=np.zeros((3,model.nv))

    mujoco.mj_jacBody(
        model,
        data,
        J_pos,
        None,
        ee_id
    )

    J=J_pos[:,:7]

    cvel=J@qvel # Current Cartesian velocity

    # Cartesian errors
    c_error = cpos_d - cpos
    cvel_error = cvel_d - cvel

    # Store data
    actual_positions.append(cpos.copy())
    desired_positions.append(cpos_d)
    tracking_errors.append(np.linalg.norm(c_error))
    times.append(t)

    # Cartesian PD gains
    Kp = np.diag([150,150,150])
    Kd = np.diag([30,30,30])

    force = Kp@c_error + Kd@cvel_error # Compute force
    tau = J.T@force # Compute torque command  

    # Observation vector given to PPO
    obs=np.concatenate([
        cpos,
        cvel,
        c_error
    ]).astype(np.float32)

    # Predict residual torque corrections from PPO policy
    action,_=model_rl.predict(
        obs,
        deterministic=True
    )

    tau = tau + action # PD torque + RL torque correction

    tau=np.clip(tau,-50,50) 

    data.ctrl[0:7] = tau # Send torque commands to 7 arm actuators 
    data.ctrl[7] = 0 # Leave the gripper actuator inactive

    mujoco.mj_step(model, data) 
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

print(
    "Mean tracking error:",
    np.mean(
        tracking_errors[
            int(len(tracking_errors)*0.2):
        ]
    )
)

print(
    "Max tracking error:",
    np.max(tracking_errors)
)

# Convert logs to arrays

actual_positions=np.array(
    actual_positions
)

desired_positions=np.array(
    desired_positions
)


# Desired vs actual trajectory

plt.figure()

plt.plot(
    desired_positions[:,0],
    desired_positions[:,1],
    label="Desired"
)

plt.plot(
    actual_positions[:,0],
    actual_positions[:,1],
    label="Actual"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title(
    "Desired vs Actual Trajectory"
)

plt.legend()

plt.axis("equal")

plt.show()


# Tracking error

plt.figure()

plt.plot(
    times,
    tracking_errors
)

plt.xlabel("Time")
plt.ylabel("Tracking Error")
plt.title(
    "Tracking Error Over Time"
)

plt.show()