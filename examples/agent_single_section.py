""" 
This code shows how to use the monte carlo control algorithm 
to train a robot arm to follow a 3D line trajectory.
The robot arm is defined by its Denavit-Hartenberg matrix 
and initial joint angles. 

The robot arm environment is defined by the robot arm 
and the observation space. 

The monte carlo control agent is trained for a 
specified number of episodes
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

# Define the environment
observation_dim = [0.002, 0.004, 0.003]
env = RobotArmEnv(robot_arm, observation_dim, render_mode=None)

# Define the monte carlo control agent
n_episodes = 500
agent = MonteCarloControl(env, n_episodes)
agent.train()
agent.show_results()
agent.test()
