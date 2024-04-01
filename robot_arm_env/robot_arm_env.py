import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym

from gymnasium import spaces
from itertools import product

class RobotArmEnv(gym.Env):
    """
    A Robot Arm Environment for simulating a robotic arm with X degrees of freedom.

    This class simulates a robot arm environment where the agent controls the movement 
    of the robot arm to achieve a target pose (x,y,z,alpha,beta,gamma). 

    Args:
        robot_arm (RobotArm): 
            A reference to a `RobotArm` object representing the simulated robot arm.
            This object includes methods for pose detection and forward kinematics .

        observation_dim (list of int): 
            The x,y,z dimensionality of the robot arm's observation space in meters. 


        max_steps (int, optional): 
            The maximum number of steps allowed per episode in the environment. 
            Defaults to 5000.

        tolerances (list of float, optional): 
            A list containing two elements representing the tolerances for 
            successful goal achievement. The first element specifies the 
            allowed tolerance for the end effector's position error 
            relative to the trajectory as radius in meter. The second element specifies 
            the allowed tolerance for the end effector's orientation error in degree. 
            Defaults to [0.03, 10].

        render_mode (str, optional): 
            The rendering mode for the environment. 
            Can be one of the following options:
                * 'human' (default for some environments): Renders the environment 
                    in a way suitable for a human observer.
            Defaults to None.

        render_skip (int, optional): 
            The number of simulation steps to skip between rendering frames. 
            This can be used to reduce the rendering frequency for faster 
            visualization. Defaults to 100.

        save_location (str, optional): 
            The path to a directory where images and other data can be saved. 
            Defaults to an empty string which results in saving 
            the results in the project root directory.

    Methods:
        reset():
            Resets the environment to its initial state and returns the initial observation.

        step(action):
            Executes an action in the environment and returns the resulting observation, reward, 
            termination flag, truncation flag, and information dictionary.
    """
    
    metadata = {"render_modes": ["human",]}

    def __init__(self, 

                robot_arm,
                observation_dim,

                max_steps           = 5000,
                tolerances          = [0.03, 10],

                render_mode         = None, 
                render_skip         = 100,

                save_location       = ''):
        
        self.rob = robot_arm
        self.obs_dim = np.array(observation_dim)

        self.render_skip        = render_skip          
        self.save_location      = save_location              
        
        self.max_steps          = max_steps                                 
        self.num_steps          = 0                                        

        # Define the start and stop (xyz) positions of the trajectory
        self.start              = np.round(self.rob.get_tcp_pose(self.rob.init)[:3],self.rob.dec[0])   
        self.stop               = np.round(np.add(self.start,observation_dim), self.rob.dec[0])   

        # Define the initial pose of the agent and the target pose
        self._agent_pose        = self.rob.get_tcp_pose(self.rob.init) 
        self._target_pose       = np.array([*self.stop, *self.rob.get_tcp_pose(self.rob.init)[3:]]) 
        
        # Max tolerances for trajectory and orientation deviatons [trajectory, orientation]
        self.tolerances         = tolerances
        
        # The last orientation/trajectory deviations of the robot arm
        self.last_orientdiff = 0
        self.last_distance = self._get_deviation('target')

        # Definition of valid Spaces for actions and observations"

        # The observation space is a discrete space with a size of (x,y,z) dimensions divided by the resolution
        # To include the last value, we add 1 to the size
        self.resolution        = 10**-self.rob.dec[0]                         
        self.size              = np.round((self.obs_dim/self.resolution),self.rob.dec[0]) + 1 
        self.observation_space = spaces.Discrete(int(self.size[0]*self.size[1]*self.size[2]))

        # We have 3 actions per axis ("-0.1", "0", "+0.1") and X axis, corresponding to 3^X possible actions.
        possible_actions = 3**self.rob.num_joints
        self.action_space = spaces.Discrete(possible_actions)
        
        # Define the possible values for each axis
        self.axis_possiblities = [-0.1, 0, 0.1]

        # Generate all possible actions using itertools.product
        action_possiblities = product(self.axis_possiblities, repeat=self.rob.num_joints)
        
        # Create a dictionary with the index as key and the action as value
        self._action_to_direction = {idx: action for idx, action in enumerate(action_possiblities)}

        # Set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Create Dictionary to store data for the rendered frames
        self.episode_data = {
            "pose"      : [[],[],[],[],[],[]],
            "reward"    : [],
            "avg_reward": [],
            "deviations": [[],[],[]],
        }

        # Initialize the figure and the subplots for the render mode
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None
        self.ax5 = None
          
    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observation.

        Returns:
            observation (np.array): The initial observation of the environment.
            info (dict): An information dictionary containing details about the initial state.
        """
        
        # Define the initial pose of the agent and the target pose
        self._agent_pose    = self.rob.get_tcp_pose(self.rob.init) 
        self._target_pose   = np.array([*self.stop, *self.rob.get_tcp_pose(self.rob.init)[3:]]) 

        # Reset agent to startpostion 
        self._agent_angles  = np.array(self.rob.init)
       
        # Reset the number of steps
        self.num_steps      = 0

        # Reset last orientation and target deviation
        self.last_orientdiff = 0
        self.last_distance = self._get_deviation('target')

        # Get the new observation, reward and info
        self.observation    = self._get_obs()
        self.info           = self._get_info()
        self.reward         = 0

        if self.render_mode == "human":
            self._render_frame(False)

        return self.observation, self.info

    def step(self, action):
        """
        Executes an action in the environment and returns the resulting observation, reward, 
        termination flag,truncation flag and information dictionary.

        Args:
            action (int): The action to be performed by the agent. The action represents 
                          a combination of joint movements for the robot arm.

        Returns:
            observation (np.array): The observation of the environment after the action is taken.
            reward (float): The reward received by the agent for the action taken.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has finished prematurely.
            info (dict): An information dictionary containing details about the step, 
                         including rewards and deviations.
        """
        # Map the action (element of {0,...,729}) to the direction we walk in
        update_angles = self._action_to_direction[action]
        new_angles    = self._agent_angles + update_angles
        new_tcp_pose  = self.rob.get_tcp_pose(new_angles)

        # Check if termination or truncations requirements are met
        truncated   = False
        terminated  = False

        # Clip all angles to +/-180 degrees
        new_angles = np.clip(new_angles, -180, 180)

        # Check if the new tcp pose is within the observation bounds and check if trajectory 
        # and orientation deviation are inside the specified tolerances 0 = radius, 1 = orientation
        bound_hold = np.all(new_tcp_pose[0:3] >= self.start) and np.all(new_tcp_pose[0:3]<= self.stop) 
        traj_hold = (self._get_deviation('trajectory') < self.tolerances[0])
        ori_diff = self._get_deviation('orientation')
        ori_hold = np.all(ori_diff <= self.tolerances[1]) or np.all(ori_diff >= (360-self.tolerances[1]))      
       
        # If all conditions are met, update the agent angles and location
        if  bound_hold and traj_hold and ori_hold:
            self._agent_angles = new_angles
            self._agent_pose   = new_tcp_pose

        # Check if the episode has exceeded the maximum number of steps, else increase step count
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            self.num_steps += 1
        
        # An episode is done if the agent has reached the target
        if self._get_deviation('target') == 0:
            terminated = True
            print("Target reached! Episode terminated.")

        # Get the new observation, reward and info
        self.reward          = self._reward()
        self.observation     = self._get_obs()
        self.info            = self._get_info()

        # Render the frame if the render mode is set to human
        if self.render_mode == "human":
            end = terminated or truncated
            self._render_frame(end)
        #print(f'Action {action} Reward {self.reward} Ori. Diff {self.info["orientation_difference"]} Traj. Dist {self.info["trajectory_distance"]} Tar. Dist {self.info["target_distance"]} Pos. {self.info["tcp_position"]} Ori. {self.info["tcp_orientation"]}')

        return self.observation, self.reward, terminated, truncated, self.info
   
    def _get_deviation(self, objective):
        """Calculate the deviation between the agent location/orientation and the target or the trajectory."""
        if objective == 'target':
            deviation = np.linalg.norm(self._agent_pose[:3] - self._target_pose[:3])
            deviation = np.round(deviation,self.rob.dec[0])
        elif objective == 'trajectory':
            P = self._agent_pose[:3]    # point to calculate the distance from
            u = self.start              # starting point of the line
            v = self.stop - self.start  # vector from start to stop (slope)
            # calculate the distance between the point and the line
            deviation = np.linalg.norm(np.cross((P - u),v))/np.linalg.norm(v)
            deviation = np.round(deviation,self.rob.dec[0])
        elif objective == 'orientation':
            orientation         = self.rob.get_tcp_pose(self._agent_angles)[3:]
            deviation    = np.abs(self._target_pose[3:] - orientation)
            deviation = np.round(deviation,self.rob.dec[1])
        return deviation
    
    def _reward(self):
        """Calculate the reward for the agent based on the trajectory.
        Returns the reward as a float.
        """
        # Linear reward for moving towards the target
        curr_distance   = self._get_deviation('target')
        max_distance    = np.round(np.linalg.norm(self.start - self.stop),self.rob.dec[0])

        target_reward   = ((self.last_distance - curr_distance)/max_distance)*10
        
         # If the agent is at the target, give a reward of 5
        if curr_distance == 0:
            target_reward = 5 # = target_reward + 1

        # Punishment for each step
        step_reward = -(1 - (curr_distance/max_distance))/100

        # update last distance
        self.last_distance = curr_distance

        # Punishment for changing the orientation
        orientation_diff = self._get_deviation('orientation')

        diff_to_last = self.last_orientdiff - orientation_diff
        diff_to_last = np.max(diff_to_last)
        orientation_reward = 0
        if(diff_to_last >= 0):
            orientation_reward = 0.04
        else:
            orientation_reward = -0.08

        # Punishment for moving away from the trajectory
        distance = self._get_deviation('trajectory')
        trajectory_reward = -distance*10

        # Sum up the rewards
        total_reward = np.round((trajectory_reward + target_reward + step_reward + orientation_reward),self.rob.dec[0])
        return total_reward

    def _get_obs(self):
        """Calcualte the observation index based on the current x,y,z position of the agent."""
        pos     = self._agent_pose[:3]
        int_pos = np.round((pos - self.start) / self.resolution).astype(int)
        x_len   = int(np.round((self.stop[0] - self.start[0]) / self.resolution))
        y_len   = int(np.round((self.stop[1] - self.start[1]) / self.resolution))
        obs     = int_pos[0] + int_pos[1]*(x_len+1) + int_pos[2]*(x_len+1)*(y_len+1)  # +1 to include the last value
        return obs

    def _get_info(self):
        """Return the information dictionary for the current step."""
        return {
            "joint_angles":                 self._agent_angles,
            "tcp_position":                 self._agent_pose[:3],
            "tcp_orientation":              self._agent_pose[3:],
            "target_distance":              self._get_deviation('target'), 
            "trajectory_distance":          self._get_deviation('trajectory'),
            "orientation_difference":       self._get_deviation('orientation'),
        }
    
    def _render_frame(self, end):
        """Render the current frame of the environment."""

        # Set the base style of the plots
        plt.style.use('seaborn-v0_8-whitegrid')
       
        # Initialize the figure and the subplots
        if self.fig is None and self.render_mode == "human":
            
            # Enable interactive mode and create a figure
            plt.ion()
            self.fig = plt.figure(layout='constrained', figsize=(10, 6))

            # Define the grid layout (3 rows, 2 columns)
            gs = gridspec.GridSpec(6, 2, figure=self.fig)

            # Create subplots
            self.ax1 = self.fig.add_subplot(gs[:3, 0], projection='3d')
            self.ax2 = self.fig.add_subplot(gs[:6, 1], projection='3d')  
            self.ax3 = self.fig.add_subplot(gs[3, 0])
            self.ax4 = self.fig.add_subplot(gs[4, 0], sharex = self.ax3)
            self.ax5 = self.fig.add_subplot(gs[5, 0], sharex = self.ax3)

            # Remove duplicate x-axis (for those axis who share the x-axis with ax3)
            plt.setp(self.ax4.get_xticklabels(), visible=False)
            plt.setp(self.ax3.get_xticklabels(), visible=False)

            # Set the title of the figure
            self.fig.suptitle('Robot Arm Environment', fontsize=16)

        # Append Data to the episode data dictionary
        if self.render_mode == "human":
             # Append data to the episode data dictionary
            [self.episode_data["pose"][idx].append(item) for idx, item in enumerate(self._agent_pose)]
            self.episode_data["reward"].append(self.reward)
            self.episode_data["avg_reward"].append(np.mean(self.episode_data["reward"]))
            #[self.episode_data["deviations"][idx].append(item) for idx, item in enumerate(self.curr_deviations)]
            self.episode_data["deviations"][0].append(self._get_deviation('target'))
            self.episode_data["deviations"][1].append(self._get_deviation('trajectory'))
            self.episode_data["deviations"][2].append(np.max(self._get_deviation('orientation')))
        # Render the frame if the render mode is set to human 
        # and the current step is a multiple of the render skip
        # or if the episode has terminated or been truncated
        if (self.render_mode == "human") and (self.num_steps % self.render_skip == 0) or end:
            # Calculate the forward kinematics to get the joint locations
            joints = self.rob.get_fw_kin(self._agent_angles)
            # [subplots 1,2] Plot the robot arm and a close up of the observation space 
            for n, self.ax in enumerate([self.ax1,self.ax2]):
                # Clear the plot
                self.ax.clear()

                # Set the axes limits
                if n == 0:
                    x_l = [-0.35 , 0.25]
                    y_l = [-0.35 , 0.35]
                    z_l = [0    , 0.5]
                    self.ax1.set_title('Robot Arm')
                    self.ax1.set(xlim3d=x_l)
                    self.ax1.set(ylim3d=y_l)
                    self.ax1.set(zlim3d=z_l)
        
                elif n == 1:
                    x_l = [self.start[0], self.stop[0]]
                    y_l = [self.start[1], self.stop[1]]
                    z_l = [self.start[2], self.stop[2]]
                    self.ax2.set_title('Observation Space')

                    self.ax2.set(xlim3d=x_l, xlabel='X')
                    self.ax2.set(ylim3d=y_l, ylabel='Y')
                    self.ax2.set(zlim3d=z_l, zlabel='Z')

                # Draw the trajetory as a gray dotted line 
                t_x = [self.start[0], self.stop[0]]
                t_y = [self.start[1], self.stop[1]]
                t_z = [self.start[2], self.stop[2]]   
                self.ax.plot(t_x,t_y,t_z,label='Trajectory', 
                             color = 'gray', linewidth = 1, 
                             linestyle = 'dotted')
                
                # Draw the observation space
                # Calculate edge lengths based on corner points
                start = self.start
                edge  = self.stop - self.start

                # Define all 8 vertices of the cube based on corner points and edge lengths
                vertices = np.array([
                    start,
                    [start[0] + edge[0], start[1], start[2]],
                    [start[0] + edge[0], start[1] + edge[1], start[2]],
                    [start[0], start[1] + edge[1], start[2]],
                    [start[0], start[1], start[2] + edge[2]],
                    [start[0] + edge[0], start[1], start[2] + edge[2]],
                    [start[0] + edge[0], start[1] + edge[1], start[2] + edge[2]],
                    [start[0], start[1] + edge[1], start[2] + edge[2]],
                ])

                # Extract edges from vertices (each edge is a pair of vertices)
                edges = np.array([
                    [vertices[0], vertices[1]],
                    [vertices[1], vertices[2]],
                    [vertices[2], vertices[3]],
                    [vertices[3], vertices[0]],
                    [vertices[4], vertices[5]],
                    [vertices[5], vertices[6]],
                    [vertices[6], vertices[7]],
                    [vertices[7], vertices[4]],
                    [vertices[0], vertices[4]],  # Corrected: include bottom edges
                    [vertices[1], vertices[5]],
                    [vertices[2], vertices[6]],
                    [vertices[3], vertices[7]]
                    ])

                # Plot each edge using a separate line plot
                for edge in edges:
                    self.ax.plot(*edge.T, color='red', linewidth=1, linestyle = 'dotted', alpha=0.5)
                self.ax.plot(*edge.T[0], color='red', linewidth=1, 
                             linestyle = 'dotted', label='Obs. Space',
                             alpha=0.5)

                # Draw the robot arm
                j_x, j_y, j_z = [np.zeros(shape=(self.rob.num_joints+2)) for _ in range(3)]
                for idx, item in enumerate([j_x, j_y, j_z]):
                    item[1:8] = joints[:,idx,3]
                self.ax.plot(j_x, j_y, j_z, markerfacecolor='k', 
                             markeredgecolor='b', marker='o', 
                             markersize=5, alpha=0.7, linewidth = 4)

                # Draw the tool center point and visualize the orienation
                tcp_x, tcp_y, tcp_z = [np.zeros(shape=(4)) for _ in range(3)]

                scale = 0.05             # scale the length of the vectors
                m = self.rob.num_joints  # last joint is the end effector

                for n in range(4):
                    tcp_x[n] = joints[m, 0, n]*scale
                    tcp_y[n] = joints[m, 1, n]*scale
                    tcp_z[n] = joints[m, 2, n]*scale
                
                # Iterate through the tcp points and colors
                for n,c in zip(range(3),['r','g','b']):
                    self.ax.plot([j_x[m+1],tcp_x[n]+j_x[m+1]], 
                                    [j_y[m+1],tcp_y[n]+j_y[m+1]], 
                                    [j_z[m+1],tcp_z[n]+j_z[m+1]], 
                                    color = c, markersize=5, alpha=0.7, linewidth = 1)
                
                # Draw a line for points the end effector visited
                self.ax.plot(self.episode_data['pose'][0],
                            self.episode_data['pose'][1],
                            self.episode_data['pose'][2],
                            color = 'g', alpha=0.7, linewidth = 1)
                
                # Draw plot and flush events to enable interactive mode
                self.ax.tick_params(axis='both', which='major', labelsize=7, pad = -2.5)
                self.fig.canvas.draw() # draw new contents
                self.fig.canvas.flush_events() 

            # [subplot 3] Plot reward and average reward
            self.ax3.clear()
            self.ax3.plot(self.episode_data['reward'], 
                          label='$R$', color = 'red', 
                          marker = 'o', linewidth = 0, 
                          markersize = 1, alpha = 0.5)
            self.ax3.plot(self.episode_data['avg_reward'], 
                          label='$\overline{R}$', color = 'blue', 
                          linestyle = '--', alpha = 0.5, linewidth = 1)
            self.ax3.legend(loc = 'upper left',frameon = True, 
                            fancybox = True, shadow = True, 
                            framealpha = 1, prop={'size': 6})
            self.ax3.set_ylabel('Reward',rotation=0, 
                                va = 'center_baseline') 
            self.ax3.yaxis.set_label_coords(-0.2,0.5)
            self.ax3.set_xticks([])

            # [subplot 4] Plot graph for orientation deviation
            self.ax4.clear()
            self.ax4.plot(self.episode_data['deviations'][2], 
                          label='Orientation', color = 'red', 
                          linewidth = 1, alpha = 0.5, linestyle = '--')
            self.ax4.set_ylabel('Ori. Error\n'+r'$max(\alpha,\beta,\gamma)$',
                                rotation=0, va = 'center_baseline') 
            self.ax4.yaxis.set_label_coords(-0.2,0.5)
            self.ax4.legend(loc = 'upper left',frameon = True, 
                            fancybox = True, shadow = True, 
                            framealpha = 1, prop={'size': 6}, handlelength = 1)
            
            # [subplot 5] Plot graph for target and trajectory deviation
            self.ax5.clear()
            self.ax5.plot(self.episode_data['deviations'][0], 
                          label='$Target $', color = 'blue', 
                          linewidth = 1, alpha = 0.5, linestyle = '--')
            self.ax5.plot(self.episode_data['deviations'][1], 
                          label='$Trajectory$', color = 'green', 
                          linewidth = 1, alpha = 0.5, linestyle = '--')
            self.ax5.set_ylabel('Pos. Error \n $|(x;y;z)|$',
                                rotation=0, va = 'center_baseline') 
            self.ax5.yaxis.set_label_coords(-0.2,0.5)
            self.ax5.legend(loc = 'upper left',frameon = True, 
                            fancybox = True, shadow = True, 
                            framealpha = 1, prop={'size': 6}, handlelength = 1)
            self.ax5.set_xlabel('Steps')

            # Remove duplicate x-axis
            plt.setp(self.ax4.get_xticklabels(), visible=False)
            plt.setp(self.ax3.get_xticklabels(), visible=False)
            if end:
                self._save_frame()
               
    def _save_frame(self):
        """Save the figure as a .png file in the current directory if not specified otherwise"""
        if self.save_location != '':
            plt.savefig(f'{self.save_location}//robotarm_env_steps{len(self.episode_data["reward"])}.png')
        else:
            plt.savefig(f'robotarm_env_steps{len(self.episode_data["reward"])}.png')   


# Example usage of the RobotArmEnv class
if __name__ == "__main__":
    from robot_arm import RobotArm
    
    robot_arm   = RobotArm(dh_matrix, init_angles, dec_prec)

    observation_dim = [0.002, 0.004, 0.003]
    env = RobotArmEnv(robot_arm, observation_dim, render_mode="human")
    observation, info = env.reset()
    end = False
    
    while not end:
        action = env.action_space.sample()  # this is where you would insert your policy
        obs, reward, term, trunc, info = env.step(action)
        end = term or trunc
        print(info)
    