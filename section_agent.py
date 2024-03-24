from robot_arm_env import RobotArmEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import plasma
import matplotlib.ticker as ticker
from timer import Timer

# Initialize timer object and lists to store the results and set numpy print precision 
timer = Timer()
np.set_printoptions(precision=3, suppress=True)
steps       = []
avg_steps   = []
mse         = []
avg_mse     = []
finish      = []
target_dist = []
avg_dist    = []

# Monte Carlo Control Algorithm
def monte_carlo_control(env, num_episodes, epsilon=0.99, gamma = 0.9, Q = None,deterministic = False):
    # Initialize action-value function with a small random value to prevent ties
    if Q is None:
        Q           = np.round(-0.1*np.random.rand(env.action_space.n, env.observation_space.n),6)
    else:
        pass# Q = np.flip(Q, axis=1)

    returns     = np.zeros((env.action_space.n, env.observation_space.n))
    N           = np.zeros((env.action_space.n, env.observation_space.n))
    # iterate over the episodes
    for n in range(num_episodes):
        #epsilon = epsilon_init*(1 - decay_rate) ** n
        epsilon = np.exp((n/num_episodes)*-5)
        if epsilon < 0.1 and not deterministic:
            epsilon = 0.1
        print(f'epsilon: {epsilon}')
        # Generate an episode
        episode = []
        error   = []
        obs, _  = env.reset()
        state   = obs
        terminated  = False
        truncated   = False
        while not (terminated or truncated):
            # pick a random action out if the random number is less than epsilon
            if np.random.randn() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            # otherwise, pick the action with the highest Q value
            else:
                action = np.argmax(Q[:, state])  # Greedy policy    

            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            #print(f'State: {state}, Action: {action}, Reward: {reward} Info: {info}')
            #print(f'State: {state}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}')
            error.append(info['trajectory_distance']**2)

        # extract the target distance and the number of steps from the info dictionary and append them to the respective lists
        # 
        steps.append(len(episode))
        avg_steps.append(np.mean(steps))
        target_dist.append(info['target_distance'])
        avg_dist.append(np.mean(target_dist))

        # print general information
        print(f'Episode finished after {len(episode)} steps')
        print(f'TCP Position: {info["tcp_position"]}')

        
        if terminated:
            finish.append(1)
        elif truncated:   
            finish.append(0)

        mse.append(np.mean(error))
        print(f'Mean Squared Error: {np.mean(error)}')
        avg_mse.append(np.mean(mse))
        #print(f'Episode: {episode}')
        #if terminated:
        # Update action-value function
        G = 0 # return of the episode
        #gamma = 0.9 # discount factor
        # iterate over the episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            # sum of the rewards from the current state to the end of the episode
            G = gamma*G + reward
            # if the pair (state, action) has not been visited before, update the Q value
            # (first visit of state-action pair)
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                returns[action, state] = returns[action, state] + G
                N[action, state] = N[action, state] + 1
                Q[action, state] = returns[action, state] / N[action, state]
        print(f'Episode {n+1}/{num_episodes} complete\n')
            # normalize the Q values and round them to 3 decimal places
            #Q = np.round(Q / np.max(Q),3)
        # return the final joint angles
    final_joint_angles = info['joint_angles']
    print(f'Final Joint Angles: {final_joint_angles}')

    return Q, final_joint_angles

def plot_results(env, Q, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma):
    plt.style.use('seaborn-v0_8-whitegrid')
    # print general information
    print("Action-Value Function: \n{Q}")
    
    print(f'{np.count_nonzero(finish)} Episodes terminated, {np.count_nonzero(finish == 0)} Episodes truncated')

    # create a figure 
    fig = plt.figure(layout='constrained', figsize=(10, 6))
    title = 'Monte Carlo Control Algorithm ('+r'$\epsilon}$' + f' = {epsilon}, ' + r'$\gamma$'+ f' = {gamma})'
    fig.suptitle(title, fontsize=16)

    # [1] plot the number of steps per episode
    ax1 = fig.add_subplot(221)
    step_max = [np.argmax(steps), np.max(steps)]
    step_min = [np.argmin(steps), np.min(steps)]
    ax1.plot(steps, label='$Steps$', marker = 'o', linewidth = 0, markersize = 1)
    ax1.plot(avg_steps, linestyle='--', label='$\overline{Steps}$',linewidth = 1)
    ax1.plot(step_min[0], step_min[1], marker = 'o', markersize = 1.5, color = 'blue', label = f'$min = {step_min[1]}\,steps$', linewidth = 0)
    ax1.plot(step_max[0], step_max[1], marker = 'o', markersize = 1.5, color = 'red', label = f'$max = {step_max[1]}\,steps$', linewidth = 0)
    ax1.legend(loc = 'upper left',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
    ax1.set_title('Number of Steps per Episode')
    ax1.set_xlabel('$Episode$')
    ax1.set_ylabel('$Stepcount$')
    
    # [2] plot the mean squared error per episode
    ax2 = fig.add_subplot(222)
    mse = np.array(mse)*1000*1000
    avg_mse = np.array(avg_mse)*1000*1000
    mse_max = [np.argmax(mse), np.max(mse)]
    mse_min = [np.argmin(mse), np.min(mse)]
    ax2.plot(mse, label='$MSE$', marker = 'o', linewidth = 0, markersize = 1)
    ax2.plot(avg_mse, linestyle='--',label='$\overline{MSE}$', linewidth = 1)
    ax2.plot(mse_min[0], mse_min[1], marker = 'o', markersize = 1.5, color = 'blue', label = f'$min = {mse_min[1]:.2f}\,mm^2$', linewidth = 0)
    ax2.plot(mse_max[0], mse_max[1], marker = 'o', markersize = 1.5, color = 'red', label = f'$max = {mse_max[1]:.2f}\,mm^2$', linewidth = 0)
    ax2.legend(loc = 'upper left',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
    ax2.set_title('Mean Squared Error per Episode')
    ax2.set_xlabel('$Episode$')
    ax2.set_ylabel('$MSE\,[mm^2]$')

    # [3] plot the target distance per episode
    # scale the target distance to mm
    target_dist = np.array(target_dist)*1000
    avg_dist    = np.array(avg_dist)*1000
    max_dist = [np.argmax(target_dist), np.max(target_dist)]
    min_dist = [np.argmin(target_dist), np.min(target_dist)]
    ax3 = fig.add_subplot(223)
    ax3.plot(target_dist, label='$Distance$', marker = 'o', linewidth = 0, markersize = 1)
    ax3.plot(avg_dist, linestyle='--', label='$\overline{Distance}$', linewidth = 1)
    ax3.plot(min_dist[0], min_dist[1], marker = 'o', markersize = 1.5, color = 'blue', label = f'$min = {min_dist[1]:.2f}\,mm$', linewidth = 0)
    ax3.plot(max_dist[0], max_dist[1], marker = 'o', markersize = 1.5, color = 'red', label = f'$max = {max_dist[1]:.2f}\,mm$', linewidth = 0)
    ax3.legend(loc = 'upper left',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
    ax3.set_title('Target Distance per Episode')
    ax3.set_xlabel('$Episode$')
    ax3.set_ylabel('$Distance\,[\,mm]$')


    # [4] plot the max(Q) values
    # get the size of the environment
    size = env.size.astype(int)
    #Q = (Q - np.min(Q))/(np.max(Q) - np.min(Q))
    q_max = np.amax(Q, axis=0)
    q_max = np.reshape(q_max, size, order = 'F')
    #q_max = np.flip(q_max, axis=1)
    #q_max = np.reshape(q_max, (11, 11, 11), order = 'F')
    # create a 3d voxel plot of the data where the values are represented by the color of the voxels
    # Create the 3D voxel plot (consider using colormaps for better visualization)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_position([0.53, 0.05, 0.45, 0.35])

    # Apply the colormap
    q_bool = q_max > -100  # Define a boolean mask for data
    q_max = np.where(q_max > -100, q_max, np.max(q_max)) # replace values smaller -100 with max value to show only relevant data
    cmap = plasma  # Replace with your chosen colormap
    norm = plt.Normalize(q_max.min(), q_max.max())  # Normalize data for colormap
    colors = cmap(norm(q_max))  # Map Q-values to colormap

    # Create voxels with color
    ax4.voxels(q_bool, facecolors=colors, shade = False, alpha = 0.5)
    ax4.set_title('Max(Q) Values')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.tick_params(axis='both', which='major', labelsize=7)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #formatter = ticker.FormatStrFormatter(f"%1.2f")
    colorbar = fig.colorbar(sm, ax = ax4, pad = 0.15)
    #colorbar.set_anchor(ax4, anchor='E')
    #colorbar.set_pad(0.2)  # Set padding between colorbar and ax

    eps_str = str(epsilon).replace('.','_')
    gamma_str = str(gamma).replace('.','_')
    
    plt.savefig(f'robot-arm-env//results//mcc_episodes_{num_episodes}_epsilon_{eps_str}_gamma_{gamma_str}.png')
    plt.show()

def render_episode(env, Q):
    # Initialize action-value function with a small random value to prevent ties
    Q           = -0.1*np.random.rand(env.action_space.n, env.observation_space.n)
    returns     = np.zeros((env.action_space.n, env.observation_space.n))
    N           = np.zeros((env.action_space.n, env.observation_space.n))

    for n in range(num_episodes):
        # Generate an episode
        obs, _ = env.reset()
        state = obs
        terminated = False
        truncated = False
        epsilon = 0.99
        while not terminated or truncated:
            # pick a random action out if the random number is less than epsilon
            if np.random.randn() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            # otherwise, pick the action with the highest Q value
            else:
                action = np.argmax(Q[:, state])  # Greedy policy     
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state

if __name__ == "__main__":
    # Example usage:
    # Define all parameters of the enviroment you want to customize
    render_mode         = None
    render_skip         = 100
    max_steps           = 5000
    init_angles         = [90,  -45, -90, -135, 0, 45]
    size                = [0.008, 0.016 ,0.012]
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

    ###############################################################################     
    # split size into sections
    sections = 2
    
    size = np.array([size[0]/sections, size[1]/sections, size[2]/sections])
    size = np.round(size, dec_obs)
    print(f'#'*100)
    print(f'Size: {size}')
    Q_section = []

    # start timer
    timer.start()

    # iterate over the sections
    for sec in range(sections):

        # Print the current section
        print(f'Section {sec+1}/{sections}')

        # Initialize the Gym environment
        env = RobotArmEnv(render_mode, render_skip, 
                        max_steps, init_angles, size, 
                        target_orientation,tolerance, 
                        dec_obs, dec_act, dh_matrix) 
        
        # Print the action and observation space of the environment
        print(f'Action: {env.action_space.n}')
        print(f'Observation Space:{env.observation_space.n}')
        
        # Run Monte Carlo control algorithm
        Q, init_angles = monte_carlo_control(env, num_episodes, epsilon, gamma)

        # Deterministic Angle finding
        temp_epsilon = 0 
        temp_num_episodes = 1
        Q, init_angles = monte_carlo_control(env, temp_num_episodes, temp_epsilon, gamma, deterministic = True)
        
        # Append the Q values of the current section to the Q_section list
        Q_section.append(Q)

        # Plot results
        #plot_results(env, Q, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)
        #epsilon = 0.01

    
    # Stitch all sections together and render an episode using the learned Q-values to visualize the results
    # create final enviroment
    print(f'#'*100)
    print(f'Final Environment')
    size = size*sections
    print(f'Size: {size}')

    render_mode = 'human'
    tolerance   = 180
    env = RobotArmEnv(render_mode, render_skip, 
                    max_steps, init_angles, size, 
                    target_orientation, tolerance,
                    dec_obs, dec_act, dh_matrix) 
    
    # create a new Q matrix to store the final Q values
    Q_result = -100*np.ones((env.action_space.n, env.observation_space.n))
    
    # insert section Q values into the final Q matrix
    for sec in range(sections):
        env_dim = env.size                         # get total dimensions
        sec_size = env_dim//sections               # get section dimensions
        sec_offset = (env.size)//sections*sec      # calculate x,y,z offset of each section

        # start and end x,y,z coordinates of the current section
        start = sec_offset.astype(int)
        end = (sec_offset+sec_size+1).astype(int) # +1 to include the last element
        #print(f'Start: {start}')
        #print(f'End: {end}')
        
        # iterate over the section and insert the Q values into the final Q matrix
        for k,z in enumerate(list(range(start[2], end[2]))):
            for l,y in enumerate((list(range(start[1], end[1])))):
                for m,x in enumerate(list(range(start[0], end[0]))):
                    # index of the current state in the total environment
                    index = int(x + y*env_dim[0] + z*env_dim[0]*env_dim[1])
                    # index of the current state in the section
                    index2 = int(m + l*(sec_size[0]+1) + k*(sec_size[0]+1)*(sec_size[1]+1))
                    # do not overwrite the Q value of the first state in following sections
                    #if sec > 0 and index2 != 0:
                        # extract the Q values of the current section
                    #    Q_result[:, index] = Q_section[sec][:, index2]
                    #else:
                    # extract the Q values of the current section
                    Q_result[:, index] = Q_section[sec][:, index2]

    
    # timer stop
    elapsed_time = timer.stop()
    print(f'Elapsed time: {elapsed_time:.2f} seconds')

    # render an episode using the learned Q-values
    plot_results(env, Q_result, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)

    # final training
    #num_episodes = 100
    #Q_result, _ = monte_carlo_control(env, num_episodes, epsilon, Q = Q_result)
    #plot_results(env, Q_result, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)

    # render an episode using the learned Q-values
    epsilon = 0
    num_episodes = 1
    monte_carlo_control(env, num_episodes, epsilon, Q = Q_result)
    outfile = 'robot-arm-env//results//Q_values.npy'
    np.save(outfile, Q_result)