from stable_baselines3 import PPO
from env import FrankaTrackingEnv

env=FrankaTrackingEnv()

# Initialise PPO policy
model=PPO(
    "MlpPolicy",
    env,
    verbose=1
)

# Train policy
model.learn(
    total_timesteps=30000
)

# Save trained model
model.save(
    "franka_policy"
)