import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma

class MonteCarloControl:
    """Monte Carlo Control algorithm for training and testing an agent.
    
    Args:
        env (object): 
            The environment object.
        n_episodes (int): 
            The number of episodes to train the agent.
        Q (numpy.ndarray): 
            The action-value function to use for training.
        save_location (str): 
            The location to save the results.
    Methods:
        train(self):
            Train the agent using the Monte Carlo Control algorithm.
        test(self):
            Test the the agent using the learned policy.
    """
    
    def __init__(self, 
                 env, 
                 n_episodes, 
                 Q=None, 
                 save_location = ''):
        
        self.env           = env
        self.n_episodes    = n_episodes
        self.Q             = Q
        self.save_location = save_location

        # Initialize action-value function with a small random value to prevent ties
        if self.Q is None:
            Q = -0.1*np.random.rand(env.action_space.n, env.observation_space.n)
            self.Q = np.round(Q, 6)

        self.returns     = np.zeros((env.action_space.n, env.observation_space.n))
        self.N           = np.zeros((env.action_space.n, env.observation_space.n))

        # Initialize the evaluation data dictionary
        self.eval_data = {
            "steps"           : [],
            "avg_steps"       : [],
            "mse"             : [],
            "avg_mse"         : [],
            "target_dist"     : [],
            "avg_target_dist" : [],
            "terminated"      : []
            }
        
    def train(self):
        """Train the agent using the Monte Carlo Control algorithm.
        Returns:
            The trained action-value function and the final joint angles.
        """
        for n in range(self.n_episodes):
            # Decrease the epsilon exponentially
            epsilon = np.exp((n/self.n_episodes)*-5)
            # Cap the epsilon value at 0.01
            if epsilon < 0.01:
                epsilon = 0.01

            # Print the current episode and epsilon value
            print(f'Episode {n+1}/{self.n_episodes} - Epsilon: {epsilon}')

            # Generate an episode using the current action-value function
            episode, final_joint_angles, _ = self._generate_episode(epsilon)
            
            # Update the action values using the episode generated
            self._update_action_values(episode)
            
        return self.Q, final_joint_angles

    def test(self, render_mode = "human"):
        """Test the the agent using the learned policy.
        Args:
            render_mode (str): 
                The mode to render the environment. 
                Default is "human".
        """
        # Set the epsilon value to zero for testing
        epsilon = 0

        if render_mode == "human":
            # Set the render mode to human for testing
            self.env.render_mode = "human"

        # Generate an episode using the current policy
        _, _, terminated = self._generate_episode(epsilon)

        # Reset the render mode to None
        self.env.render_mode = "None"      

        return terminated    
        
    def _generate_episode(self, epsilon):
        """Generate an episode using the current policy."""

        # Initialize the episode
        episode       = []
        squared_error = []
        obs, _        = self.env.reset()
        state         = obs
        terminated    = False
        truncated     = False
        
        while not (terminated or truncated):
            # Select an action using the epsilon-greedy policy
            if np.random.rand() < epsilon:
                # Select a random action
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Select the action with the highest action value
                action = np.argmax(self.Q[:,state])
            
            # Take the action and observe the next state and reward
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store state, action and reward from the episode
            episode.append((state, action, reward))

            # Update the current state
            state = next_state

            # Store the squared error
            squared_error.append(info["trajectory_distance"]**2)


        # Update the evaluation data
        self._update_eval_data(info, episode, squared_error, terminated)

        # Print the episode information
        print(f'Episode complete - Steps: {len(episode)}')
        print(f'TCP Position: {info["tcp_position"]}')

        # Aquire and print the final joint angles
        final_joint_angles = info["joint_angles"]
        print(f'Final Joint Angles: {final_joint_angles}')

        return episode, final_joint_angles, terminated
    
    def _update_eval_data(self, info, episode, squared_error, terminated):
        """Update the evaluation data dictionary."""

        # Store the number of steps taken in the episode
        steps = len(episode)
        self.eval_data["steps"].append(steps)

        # Store the average number of steps taken
        avg_steps = np.mean(self.eval_data["steps"])
        self.eval_data["avg_steps"].append(avg_steps)

        # Store the target distance
        target_dist = info["target_distance"]
        self.eval_data["target_dist"].append(target_dist)

        # Store the average target distance
        avg_target_dist = np.mean(self.eval_data["target_dist"])
        self.eval_data["avg_target_dist"].append(avg_target_dist)

        # Calculate the mean squared error
        mse = np.mean(squared_error)
        self.eval_data["mse"].append(mse)

        # Calculate the average mean squared error
        avg_mse = np.mean(self.eval_data["mse"])
        self.eval_data["avg_mse"].append(avg_mse)

        # Store the terminated flag
        if terminated:
            self.eval_data["terminated"].append(1)
        else:
            self.eval_data["terminated"].append(0)

    def _update_action_values(self, episode):
        """Update the action values using the episode generated."""
        # Gamma value for the discount factor
        gamma = 0.99
        # Initialize the return
        G = 0
        # Loop through the episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma*G + reward
            # Check if the state-action pair has been visited before
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                self.returns[action, state] = self.returns[action, state] + G
                self.N[action, state] = self.N[action, state] + 1
                self.Q[action, state] = self.returns[action, state] / self.N[action, state]

        print(f'Evaluation complete\n')

    def show_results(self):
        """Display the results of the training and testing phases."""

        # Set the base style of the plots
        plt.style.use('seaborn-v0_8-whitegrid')

        # Print Action-Value function
        print(f'Action-Value Function:\n{self.Q}\n')

        # Get the number of episodes that were terminated and truncated
        finished = np.count_nonzero(self.eval_data["terminated"])
        not_finished = self.n_episodes - finished
        print(f'{finished} Episodes terminated, {not_finished} Episodes truncated')

        # create a figure 
        fig = plt.figure(layout='constrained', figsize=(10, 6))
        title = 'Monte Carlo Control Algorithm'
        fig.suptitle(title, fontsize=16)

        # [1] plot the number of steps per episode
        ax1       = fig.add_subplot(221)
        steps     = np.array(self.eval_data["steps"])
        avg_steps = np.array(self.eval_data["avg_steps"])
        step_max  = [np.argmax(steps), np.max(steps)]
        step_min  = [np.argmin(steps), np.min(steps)]
        ax1.plot(steps, label='$Steps$', marker = 'o', 
                 linewidth = 0, markersize = 1)
        ax1.plot(avg_steps, linestyle='--', 
                 label='$\overline{Steps}$',linewidth = 1)
        ax1.plot(step_min[0], step_min[1], marker = 'o', 
                 markersize = 1.5, color = 'blue', 
                 label = f'$min = {step_min[1]}\,steps$', 
                 linewidth = 0)
        ax1.plot(step_max[0], step_max[1], marker = 'o', 
                 markersize = 1.5, color = 'red', 
                 label = f'$max = {step_max[1]}\,steps$', 
                 linewidth = 0)
        ax1.legend(loc = 'upper left',frameon = True, 
                   fancybox = True, shadow = True, 
                   framealpha = 1, prop={'size': 6}, 
                   ncols = 2, handlelength = 1,columnspacing = 0.25)
        ax1.set_title('Number of Steps per Episode')
        ax1.set_xlabel('$Episode$')
        ax1.set_ylabel('$Stepcount$')
        
        # [2] plot the mean squared error per episode
        ax2     = fig.add_subplot(222)
        mse     = np.array(self.eval_data['mse'])*1000*1000
        avg_mse = np.array(self.eval_data['avg_mse'])*1000*1000
        mse_max = [np.argmax(mse), np.max(mse)]
        mse_min = [np.argmin(mse), np.min(mse)]
        ax2.plot(mse, label='$MSE$', marker = 'o', linewidth = 0, markersize = 1)
        ax2.plot(avg_mse, linestyle='--',label='$\overline{MSE}$', linewidth = 1)
        ax2.plot(mse_min[0], mse_min[1], marker = 'o', 
                 markersize = 1.5, color = 'blue', 
                 label = f'$min = {mse_min[1]:.2f}\,mm^2$', 
                 linewidth = 0)
        ax2.plot(mse_max[0], mse_max[1], marker = 'o', 
                 markersize = 1.5, color = 'red', 
                 label = f'$max = {mse_max[1]:.2f}\,mm^2$', linewidth = 0)
        ax2.legend(loc = 'upper left',frameon = True, 
                   fancybox = True, shadow = True, 
                   framealpha = 1, prop={'size': 6}, 
                   ncols = 2, handlelength = 1,columnspacing = 0.25)
        ax2.set_title('Mean Squared Error per Episode')
        ax2.set_xlabel('$Episode$')
        ax2.set_ylabel('$MSE\,[mm^2]$')

        # [3] plot the target distance per episode
        # scale the target distance to mm
        ax3         = fig.add_subplot(223)
        target_dist = np.array(self.eval_data['target_dist'])*1000
        avg_dist    = np.array(self.eval_data['avg_target_dist'])*1000
        max_dist    = [np.argmax(target_dist), np.max(target_dist)]
        min_dist    = [np.argmin(target_dist), np.min(target_dist)]
        ax3.plot(target_dist, label='$Distance$', marker = 'o', 
                 linewidth = 0, markersize = 1)
        ax3.plot(avg_dist, linestyle='--', label='$\overline{Distance}$', linewidth = 1)
        ax3.plot(min_dist[0], min_dist[1], marker = 'o', 
                 markersize = 1.5, color = 'blue', 
                 label = f'$min = {min_dist[1]:.2f}\,mm$', linewidth = 0)
        ax3.plot(max_dist[0], max_dist[1], marker = 'o', 
                 markersize = 1.5, color = 'red', 
                 label = f'$max = {max_dist[1]:.2f}\,mm$', linewidth = 0)
        ax3.legend(loc = 'upper left',frameon = True, 
                   fancybox = True, shadow = True, 
                   framealpha = 1, prop={'size': 6}, 
                   ncols = 2, handlelength = 1,columnspacing = 0.25)
        ax3.set_title('Target Distance per Episode')
        ax3.set_xlabel('$Episode$')
        ax3.set_ylabel('$Distance\,[\,mm]$')


        # [4] plot the max(Q) values
        # Get the size of the environment
        size = self.env.size.astype(int)

        # Get the max(Q) values and reshape them to the size of the environment
        q_max = np.amax(self.Q, axis=0)
        q_max = np.reshape(q_max, size, order = 'F')
        
        # Create a 3d voxel plot of the data where the values 
        # are represented by the color of the voxels
        ax4 = fig.add_subplot(224, projection='3d')
        ax4.set_position([0.53, 0.05, 0.45, 0.35])

        # Define a boolean mask for data
        q_bool = q_max > -100  

        # Replace values smaller -100 with max value to show only relevant data
        q_max = np.where(q_max > -100, q_max, np.max(q_max))

        # Normalize the data
        norm = plt.Normalize(q_max.min(), q_max.max()) 

        # Map Q-values to colormap
        cmap = plasma
        colors = cmap(norm(q_max)) 

        # Create voxels with color
        ax4.voxels(q_bool, facecolors=colors, shade = False, alpha = 0.5)
        ax4.set_title('Max(Q) Values')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.tick_params(axis='both', which='major', labelsize=7)

        # Add colorbar to the plot
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax = ax4, pad = 0.15)

        # Save the figure
        if self.save_location == '':
            plt.savefig(f'mcc_{self.n_episodes}_episodes.png')
        else :
            plt.savefig(f'{self.save_location}//mcc_{self.n_episodes}_episodes.png')

        # Show the plot
        plt.show()


# Example of using the Monte Carlo Control algorithmS
if __name__ == "__main__":
    from robot_arm_env.robot_arm_env import RobotArmEnv
    from robot_arm_env.robot_arm import RobotArm
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
    n_episodes = 1
    agent = MonteCarloControl(env, n_episodes)
    agent.train()
    agent.show_results()
    agent.test()
    
    