""" 
This code shows how to load a learned a policy from a saved file and test it.
"""
from robot_arm_env.robot_arm_env import RobotArmEnv
from robot_arm_env.robot_arm import RobotArm
from rl_algorithms.monte_carlo_control import MonteCarloControl

import numpy as np

np.set_printoptions(precision=2, suppress=True)
dh_matrix = np.array([  
                    [0        ,0.15185        ,90      ,0],
                    [-0.24355 ,0              ,0       ,0],
                    [-0.2132  ,0              ,0       ,0],
                    [0        ,0.13105        ,90      ,0],
                    [0        ,0.08535        ,-90     ,0],
                    [0        ,0.0921         ,0       ,0]
                    ])
init_angles = [90,  -45, -90, -135, 0, 45]
dec_prec    = [3,1]
robot_arm   = RobotArm(dh_matrix, init_angles, dec_prec)

# Define the environment and ensure that the specified observations dimensions match the input data
observation_dim = [0.002, 0.004, 0.003]

# Scale Observation Dimenions according to Sections
n_sections = 2
observation_dim = np.array(observation_dim) * n_sections
render_skip = 1
env             = RobotArmEnv(robot_arm, observation_dim, render_mode=None, render_skip=render_skip)


# load the Q matrix from the saved file
save_file = 'example_results//2_sections_Q_values_prepared.npy'
Q         = np.load(save_file)

# Define the monte carlo control agent
n_episodes = 1
agent      = MonteCarloControl(env, n_episodes,Q)
agent.test()
