from robot_arm_env import RobotArmEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import plasma

np.set_printoptions(precision=3, suppress=True)
steps       = []
avg_steps   = []
mse         = []
avg_mse     = []
finish      = []
target_dist = []
avg_dist    = []

# Monte Carlo Control Algorithm
def monte_carlo_control(env, num_episodes, epsilon=0.01):
    # Initialize action-value function with a small random value to prevent ties
    Q           = np.round(-0.1*np.random.rand(env.action_space.n, env.observation_space.n),6)
    returns     = np.zeros((env.action_space.n, env.observation_space.n))
    N           = np.zeros((env.action_space.n, env.observation_space.n))

    for n in range(num_episodes):
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
        print(f'Target Distance: {info["target_distance"]}')
        
        if terminated:
            finish.append(1)
        elif truncated:   
            finish.append(0)

        mse.append(np.mean(error))
        print(f'Mean Squared Error: {np.mean(error)}')
        avg_mse.append(np.mean(mse))
        #print(f'Episode: {episode}')

        # Update action-value function
        G = 0 # return of the episode
        gamma = 0.9 # discount factor
        # iterate over the episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            # sum of the rewards from the current state to the end of the episode
            G = gamma*G + reward
            # if the pair (state, action) has not been visited before, update the Q value
            # (first visit of state-action pair)
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                returns[action, state] = returns[action, state] +G
                N[action, state] = N[action, state] + 1
                Q[action, state] = returns[action, state] / N[action, state]
        print(f'Episode {n+1}/{num_episodes} complete\n')
        # normalize the Q values and round them to 3 decimal places
        #Q = np.round(Q / np.max(Q),3)
    return Q

def plot_results():
    # print general information
    print("Action-Value Function: \n{Q}")
    print(f'{sum(finish)} Episodes terminated, {num_episodes-sum(finish)} Episodes truncated')

    # create a figure 
    fig= plt.figure(tight_layout=True)

    # plot the number of steps per episode
    ax1 = fig.add_subplot(221)
    ax1.plot(steps, label='Steps per Episode', marker = 'o', linewidth = 0)
    ax1.plot(avg_steps, linestyle='--', label='Average Steps per Episode')
    ax1.set_title('Number of Steps per Episode')

    # plot the mean squared error per episode
    ax2 = fig.add_subplot(222)
    ax2.plot(mse, label='Mean Squared Error per Episode', marker = 'o', linewidth = 0)
    ax2.plot(avg_mse, linestyle='--',label='Average Mean Squared Error per Episode')
    ax2.set_title('Mean Squared Error per Episode')

    # plot the target distance per episode
    ax3 = fig.add_subplot(223)
    ax3.plot(target_dist, label='Target Distance per Episode', marker = 'o', linewidth = 0)
    ax3.set_title('Target Distance per Episode')
    ax3.plot(avg_dist, linestyle='--', label='Average Target Distance per Episode')

    plt.legend()

    # plot the max(Q) values
    q_max = np.amax(Q, axis=0)
    #q_max = np.reshape(q_max, (41, 71, 61), order = 'F')
    q_max = np.reshape(q_max, (5, 5, 5), order = 'F')
    # create a 3d voxel plot of the data where the values are represented by the color of the voxels
    # Create the 3D voxel plot (consider using colormaps for better visualization)
    ax4 = fig.add_subplot(224, projection='3d')

    # Apply the colormap
    cmap = plasma  # Replace with your chosen colormap
    norm = plt.Normalize(q_max.min(), q_max.max())  # Normalize data for colormap
    colors = cmap(norm(q_max))  # Map Q-values to colormap

    # Create voxels with color
    q_bool = q_max > 0.00001  # Define a boolean mask for data
    ax4.voxels(q_max, facecolors=colors, shade = False, alpha = 0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, label="Maximum Q-Value", ax = ax4)

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
        epsilon = 0.01
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
    env = RobotArmEnv(render_mode=None) # Initialize your Gym environment
    print(env.observation_space.n)
    num_episodes = 100                   # Number of episodes for Monte Carlo control

    # Run Monte Carlo control algorithm
    Q = monte_carlo_control(env, num_episodes)
    pd.DataFrame(Q).to_csv('Q.csv')

    # Plot results
    plot_results()

    # Render an episode using the learned Q-values
    env = RobotArmEnv(render_mode='human') # Initialize your Gym environment
    render_episode(env, Q)
   