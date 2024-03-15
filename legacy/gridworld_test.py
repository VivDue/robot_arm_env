import gym
import gym_examples
from gym_examples.envs.robot_arm import RobotArmEnv
env = gym.make('gym_examples/RobotArm-v0', render_mode = 'human')
observation, info = env.reset()

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
