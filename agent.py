from robot_arm_env import RobotArmEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import plasma
import matplotlib.ticker as ticker

np.set_printoptions(precision=3, suppress=True)
steps       = []
avg_steps   = []
mse         = []
avg_mse     = []
finish      = []
target_dist = []
avg_dist    = []

# Monte Carlo Control Algorithm
def monte_carlo_control(env, num_episodes, epsilon=0.99, gamma = 0.9, Q = None):
    # Initialize action-value function with a small random value to prevent ties
    if Q is None:
        Q           = np.round(-0.1*np.random.rand(env.action_space.n, env.observation_space.n),6)
    returns     = np.zeros((env.action_space.n, env.observation_space.n))
    N           = np.zeros((env.action_space.n, env.observation_space.n))

    for n in range(num_episodes):
        epsilon = (epsilon/num_episodes)*n
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

        
        if terminated:
            finish.append(1)
        elif truncated:   
            finish.append(0)

        mse.append(np.mean(error))
        print(f'Mean Squared Error: {np.mean(error)}')
        avg_mse.append(np.mean(mse))
        #print(f'Episode: {episode}')
        if terminated:
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
    return Q

def plot_results(env, Q, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma):
    plt.style.use('seaborn-v0_8-whitegrid')
    # print general information
    print("Action-Value Function: \n{Q}")
    print(f'{sum(finish)} Episodes terminated, {num_episodes-sum(finish)} Episodes truncated')

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
    ax1.legend(loc = 'upper right',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
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
    ax2.legend(loc = 'upper right',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
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
    ax3.legend(loc = 'upper right',frameon = True, fancybox = True, shadow = True, framealpha = 1, prop={'size': 6}, ncols = 2, handlelength = 1,columnspacing = 0.25)
    ax3.set_title('Target Distance per Episode')
    ax3.set_xlabel('$Episode$')
    ax3.set_ylabel('$Distance\,[\,mm]$')


    # [4] plot the max(Q) values
    # get the size of the environment
    size = env.size.astype(int)
    #Q = (Q - np.min(Q))/(np.max(Q) - np.min(Q))
    q_max = np.amax(Q, axis=0)
    q_max = np.reshape(q_max, size, order = 'F')
    #q_max = np.reshape(q_max, (11, 11, 11), order = 'F')
    # create a 3d voxel plot of the data where the values are represented by the color of the voxels
    # Create the 3D voxel plot (consider using colormaps for better visualization)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_position([0.53, 0.05, 0.45, 0.35])

    # Apply the colormap
    cmap = plasma  # Replace with your chosen colormap
    norm = plt.Normalize(q_max.min(), q_max.max())  # Normalize data for colormap
    colors = cmap(norm(q_max))  # Map Q-values to colormap

    # Create voxels with color
    q_bool = q_max > 0.00001  # Define a boolean mask for data
    ax4.voxels(q_max, facecolors=colors, shade = False, alpha = 0.5)
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
    size                = [0.002, 0.004 ,0.003]
    target_orientation  = [0,0,0]
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
    num_episodes    = 1000
    epsilon         = 1
    gamma           = 0.5


    # Initialize the Gym environment
    env = RobotArmEnv(render_mode, render_skip, 
                      max_steps, init_angles, size, 
                      target_orientation, 
                      dec_obs, dec_act, dh_matrix) 
    
    # Print the action and observation space of the environment
    print(f'Action: {env.action_space.n}')
    print(f'Observation Space:{env.observation_space.n}')

    # Run Monte Carlo control algorithm
    Q = monte_carlo_control(env, num_episodes, epsilon, gamma)
    np.savetxt("Q.csv", Q, delimiter=",")

    # Plot results
    plot_results(env, Q, target_dist, avg_dist, mse, avg_mse, steps, avg_steps, epsilon, gamma)
    epsilon = 0.01

    # Render an episode using the learned Q-values to visualize the results
    render_mode = 'human'
    num_episodes = 1
    env = RobotArmEnv(render_mode, render_skip, 
                      max_steps, init_angles, size, 
                      target_orientation, 
                      dec_obs, dec_act, dh_matrix) 
    render_episode(env, Q)
    monte_carlo_control(env, num_episodes, Q = Q)
   