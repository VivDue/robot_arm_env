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

save_location = 'examples//example_results'

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

# Define the number of sections for the robot arm
n_sections = 2

# Initialize the list to store the Q values of each section
Q_section = []

# Initialize the initial start angle for the first section
start_angles = init_angles

# iterate over the sections
for sec in range(n_sections):

    # Print the current section
    print('#'*100 + f'\n Section {sec+1}/{n_sections} \n' +'#'*100 + '\n')

    # Define the robot arm for the current section (change init angles)
    robot_arm   = RobotArm(dh_matrix, start_angles, dec_prec)

    # Initialize the Gym environment
    env = RobotArmEnv(robot_arm, observation_dim, render_mode=None)
    
    # Run Monte Carlo control algorithm
    agent = MonteCarloControl(env, n_episodes)
    Q, result_angles = agent.train()

    # Update the initial start angle for the next section
    start_angles = result_angles
    
    # Append the Q values of the current section to the Q_section list
    Q_section.append(Q)

# Stitch all sections together and render an episode using the learned Q-values to visualize the results
# create final enviroment
print(f'#'*100 + '\n Stitched Sections \n' + '#'*100)
observation_dim = np.array(observation_dim)*n_sections

render_mode = 'human'
# Define the final robot arm
robot_arm = RobotArm(dh_matrix, init_angles, dec_prec)

# Adjust the tolerance for the angle to angle incompatablieties 
# between different sections
tolerances   = [0.03, 180]
env = RobotArmEnv(robot_arm, observation_dim, tolerances=tolerances, render_mode="human", save_location=save_location)

# Create a new Q matrix to store the final Q values and initialize it with highly negative random values 
# to simplify the visualization of the results (Q values smaller -100 will not be displayed)
Q_result = np.round(-0.1*np.random.rand(env.action_space.n, env.observation_space.n),6) -100

# Insert section Q values into the final Q matrix
for sec in range(n_sections):
    env_dim = env.size                         # get total dimensions
    sec_size = env_dim//n_sections               # get section dimensions
    sec_offset = (env.size)//n_sections*sec      # calculate x,y,z offset of each section

    # Start and end x,y,z coordinates of the current section
    start = sec_offset.astype(int)
    end = (sec_offset+sec_size+1).astype(int) # +1 to include the last element
    
    # Iterate over the section and insert the Q values into the final Q matrix
    for k,z in enumerate(list(range(start[2], end[2]))):
        for l,y in enumerate((list(range(start[1], end[1])))):
            for m,x in enumerate(list(range(start[0], end[0]))):
                # index of the current state in the total environment
                idx_1 = int(x + y*env_dim[0] + z*env_dim[0]*env_dim[1])
                # index of the current state in the section
                idx_2 = int(m + l*(sec_size[0]+1) + k*(sec_size[0]+1)*(sec_size[1]+1))
                # insert the Q values of the current section into the final Q matrix
                Q_result[:, idx_1] = Q_section[sec][:, idx_2]

outfile = f'{save_location}//{n_sections}_sections_Q_values.npy'
np.save(outfile, Q_result)

# Define the final monte carlo control agent
n_episodes = 1
agent = MonteCarloControl(env, n_episodes, Q_result)

# Test the agent
agent.test()
