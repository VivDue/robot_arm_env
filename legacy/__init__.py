from .robot_arm_env import RobotArmEnv
from gymnasium.envs.registration import register

register(
    id="RobotArm-v0",
    entry_point="robot_arm_env:RobotArmEnv",
    max_episode_steps=10000,
)
