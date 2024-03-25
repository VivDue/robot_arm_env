from section_agent import monte_carlo_control, plot_results
from robot_arm_env.robot_arm_env import RobotArmEnv
import numpy as np

# Initialize timer object and lists to store the results and set numpy print precision 
np.set_printoptions(precision=3, suppress=True)
steps       = []
avg_steps   = []
mse         = []
avg_mse     = []
finish      = []
target_dist = []
avg_dist    = []

if __name__ == "__main__":
    # Example usage:
    # Define all parameters of the enviroment you want to customize
    render_mode         = None
    render_skip         = 100
    max_steps           = 50000
    init_angles         = [90,  -45, -90, -135, 0, 45]
    #size                = [0.004, 0.008 ,0.006] 
    size                = [0.040 ,0.080 ,0.060]
    target_orientation  = [0,0,0]
    tolerance           = 10
    dec_obs             = 3
    dec_act             = 1
    dh_matrix           = np.array([
                        [0        ,0.15185        ,90      ,0],
                        [-0.24355 ,0              ,0       ,0],
                        [-0.2132  ,0              ,0       ,0],
                        [0        ,0.13105        ,90      ,0],
                        [0        ,0.08535        ,-90     ,0],
                        [0        ,0.0921         ,0       ,0]
                        ])
    
    # Define the number of episodes for Monte Carlo control
    num_episodes    = 500
    epsilon         = 1
    gamma           = 0.99

    render_mode = 'human'
    tolerance   = 180
    env = RobotArmEnv(render_mode, render_skip, 
                    max_steps, init_angles, size, 
                    target_orientation, tolerance,
                    dec_obs, dec_act, dh_matrix) 
    
    # create a new Q matrix to store the final Q values
    input_file = 'robot-arm-env//results//20_sections_5000_Episodes_Testrun_3//Q_values_19_stitched.npy'
    Q_result = np.load(input_file)
    #print(Q_result)
  
    

    # final training
    #num_episodes = 100
    #Q_result, _ = monte_carlo_control(env, num_episodes, epsilon, Q = Q_result)
    #plot_results(env, Q_result, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)

    # render an episode using the learned Q-values
    epsilon = 0.1
    num_episodes = 1
    monte_carlo_control(env, num_episodes, epsilon, Q = Q_result, deterministic = False)

    # render an episode using the learned Q-values
    plot_results(env, Q_result, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)
   