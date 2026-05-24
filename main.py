import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model,data) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # Read current arm joint positions and velocities
    # Use only the first 7 values (arm joints) and ignore the gripper
    qpos = data.qpos[:7]
    qvel = data.qvel[:7]

    t = data.time # Simulation time used to generate trajectories
    
    # Position gain matrix
    Kp = np.diag([100,100,100,80,80,50,50])

    # Damping gain matrix
    Kd = np.diag([20,20,20,15,15,10,10])

    # Desired joint-space trajectories
    qpos_d = np.array([
        0.5*np.sin(t),
        0.3*np.cos(t),
        0.2*np.sin(t),
        0.15*np.cos(t),
        0.1*np.sin(t),
        0.1*np.cos(t),
        0.05*np.sin(t)
    ])

    # Desired joint velocities
    qvel_d = np.array([
        0.5*np.cos(t),
        -0.3*np.sin(t),
        0.2*np.cos(t),
        -0.15*np.sin(t),
        0.1*np.cos(t),
        -0.1*np.sin(t),
        0.05*np.cos(t)
    ])
    
    
    # Position error
    q_error = qpos_d - qpos

    # Velocity error
    qd_error = qvel_d - qvel

    # Compute torque command  
    tau = Kp@q_error + Kd@qd_error
  
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