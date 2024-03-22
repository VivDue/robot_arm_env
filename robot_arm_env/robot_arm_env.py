import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from gymnasium import spaces
from itertools import product


class RobotArmEnv(gym.Env):
    metadata = {"render_modes": ["human",]}

    def __init__(self, 
                render_mode=None, 
                render_skip=20,
                max_steps           = 1000000,
                #trajectory_type     = 'line',
                init_angles         = [90,  -45, -90, -135, 0, 45],
                #size                = [0.04, np.sqrt(0.0048),0.06],
                #size                = [0.04, 0.07 ,0.06],
                size                = [0.010, 0.010,0.010],
                offset              = [0.000,0.000,0.000], 
                resolution          = 0.001,

                #num_waypoints       = 5, 
                target_orientation  = [0,0,0],
                dh_matrix           = np.array([
                                    [0        ,0.15185        ,90      ,0],
                                    [-0.24355 ,0              ,0       ,0],
                                    [-0.2132  ,0              ,0       ,0],
                                    [0        ,0.13105        ,90      ,0],
                                    [0        ,0.08535        ,-90     ,0],
                                    [0        ,0.0921         ,0       ,0]
                                    ])):
        self.render_skip        = render_skip                      # The number of steps to skip before rendering the next frame

        self.max_steps          = max_steps                          # The maximum number of steps for an episode
        self.num_steps          = 0                                  # The current number of steps for an episode

        self.init_angles        = np.array(init_angles)             # The initial joint angles of the robot arm                  
        #self.reset_angles       = self.init_angles                # The initial joint angles of the robot arm                  
        self.size               = np.round(np.array(size),3)        # The x,y,z dimensions of the voxel grid
        self.offset             = np.array(offset)                  # The x,y,z offset of the voxel grid
        self.resolution         = resolution                        # The resolution of the voxel grid
        self.dh_matrix          = dh_matrix                         # The Denavit-Hartenberg matrix

        #self.trajectory_type    = trajectory_type                   # The type of trajectory the robot arm should follow
        self.start              = self._tcp_pose(init_angles)[:3]   # The start position of the trajectory
        self.stop               = np.add(self.start,self.size)            # The stop position of the trajectory
        #self.num_waypoints      = num_waypoints                     # The number of waypoints for the trajectory
        #self.trajectory         = self._trajectory()                # The trajectory of the robot arm
        self._target_location   = self.stop                         # The target location of the robot arm

        self.target_orientation = target_orientation                # The orientation of the robot arm
        # Lower and upper bounds for the position and orientation of the toolhead
        self.lower_bound = np.round(self.start - self.offset,3)
        self.upper_bound = np.round(self.stop + self.offset,3)

        # Definition of valid Spaces for actions and observations"
        # Position and Orientation of the Toolhead inside a 3D box with the size and offset defined above
        
        size = np.round((self.size+self.offset*2)/self.resolution) + 1        # + 1 to include the last value
        self.observation_space = spaces.Discrete(int(size[0]*size[1]*size[2]))

        """self.observation_space = spaces.Dict(
            {
                "position":     spaces.Box(low=self.lower_bound, high=self.upper_bound, dtype=np.float32),
                "orientation":  spaces.Box(-180, 180, shape=(3,), dtype=np.float32),
            }
        )"""
        
        # We have 3 actions per axis ("-0.1", "0", "+0.1") and 6 axis, corresponding to 3^6 = 729 possible actions.
        self.action_space = spaces.Discrete(729)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the movement of all 6 axis of the robot arm.
        """ 
        
        # Define the possible values for each axis
        axis_possiblities = [-0.2, 0, 0.2]

        # Generate all possible actions using itertools.product
        action_possiblities = product(axis_possiblities, repeat=6)
        
        
        # Create a dictionary with the index as key and the action as value
        self._action_to_direction = {idx: action for idx, action in enumerate(action_possiblities)}

        # Set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.fig` will be a reference
        to the window that we draw to. 
        It will remain `None` until human-mode is used for the
        first time.
        """
        self.fig = None
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set the TCP (Tool Center Point) start position 
        self._agent_location    = self.start
        self._target_location   = self.stop

        # current waypoint on the trajectory
        #self._current_waypoint = 1
        #print(f"Waypoint {self._current_waypoint}/{len(self.trajectory)} reached")

        # reset agent to startpostion 
        self._agent_angles = np.array(self.init_angles)
        #self._agent_angles = np.array(self.reset_angles)

        # Reset the number of steps
        self.num_steps      = 0

        observation = self._get_obs()
        info = self._get_info()

        self.last_distance = self._distance('target')

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,...,729}) to the direction we walk in
        self.update_angles  = self._action_to_direction[action]
        new_angles          = self._agent_angles + self.update_angles
        new_tcp_pose        = self._tcp_pose(new_angles)

        # check if termination or truncations requirements are met
        truncated = False
        terminated = False

        # check if the TCP orientation close to the target orientation
        """orientation         = self._tcp_pose(self._agent_angles)[3:]
        orientation_diff    = np.linalg.norm(self.target_orientation - orientation)
        if orientation_diff < 0.1:
            self.reset()"""

        # clip all angles to +/-180 degrees
        #if np.any(new_angles > 180) or np.any(new_angles < -180):
        new_angles = np.clip(new_angles, -180, 180)
        #print("Joint angles out of bounds! Angles clipped.")

        # check if the new tcp pose is within the bounds and update the angles
        if np.all(new_tcp_pose[0:3] >= self.lower_bound) and np.all(new_tcp_pose[0:3] <= self.upper_bound) and self._distance('trajectory') < 0.1:
            self._agent_angles = new_angles
            self._agent_location = new_tcp_pose[:3]
        else:
            self._agent_angles = self._agent_angles
            self._agent_location = self._agent_location
            #truncated = True
            #print(f'{(new_tcp_pose[0:3] >= self.lower_bound)} - {np.all(new_tcp_pose[0:3] >= self.lower_bound)}')
            #print(f'TCP out of bounds! Position Reset.Agent location: {new_tcp_pose[0:3]} Boundaries: {self.lower_bound} - {self.upper_bound}')
            #self.reset()
             
        # Check if the episode has exceeded the maximum number of steps, else increase step count
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            self.num_steps += 1
        
        """if :
            truncated = True
            self.reset()
            print("Trajectory exceeded! Episode truncated.")"""

        # An episode is done if the agent has reached the target
        # if self._current_waypoint == len(self.trajectory):
        if self._distance('target') <= 0.001:
            terminated = True
            print("Target reached! Episode terminated.")

        # Get the new observation, reward and info
        reward          = self._reward()
        observation     = self._get_obs()
        info            = self._get_info()
    
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info
    
    def _distance(self,objective):
        """Calculate the distance between the agent and the trajectory or the target."""
        if objective == 'target':
            distance = np.linalg.norm(self._agent_location - self._target_location)
        elif objective == 'trajectory':
            P = self._agent_location    # point to calculate the distance from
            u = self.start              # starting point of the line
            v = self.stop - self.start  # vector from start to stop (slope)
            # calculate the distance between the point and the line
            distance = np.linalg.norm(np.cross((P - u),v))/np.linalg.norm(v)
        distance = np.round(distance,3)
        return distance

        
    def _reward(self):
        """Calculate the reward for the agent based on the trajectory.
        Returns the reward as a float.
        """
        # current target
        #self._target_location = self.trajectory[self._current_waypoint-1] 
        
        # Calculate the distance between the agent and the target (current waypoint on the trajectory)
        #distance = np.linalg.norm(self._agent_location - self._target_location)

        # Reward for getting closer to the target (max reward = 1)
        #max_distance = np.linalg.norm(self.start - self.stop)
        #curr_distance = self._distance('target')
        #target_reward = (1 - curr_distance/max_distance)
        #target_reward = np.round(1/distance*0.0001,3)

        # compare if current distance is smaller than the last distance
        curr_distance   = self._distance('target')
        max_distance    = np.round(np.linalg.norm(self.start - self.stop),3)
        target_reward   = ((self.last_distance - curr_distance)/max_distance)

        # negative reward for each step
        step_reward = -0.001

        # update last distance
        self.last_distance = curr_distance

        # Punishment (Negative Reward) for moving away from the trajectory
        distance = self._distance('trajectory')
        trajectory_reward = -distance*10
        
        """# Reward for progress along the trajectory
        progress_reward = 0
        
        # If the agent is close to the target and hasn't reached it, move to the next target
        if distance < 0.003 and self._current_waypoint < len(self.trajectory):
            self._current_waypoint += 1
            progress_reward = 1
            self.reset_angles = self._agent_angles
            #print(f"Waypoint {self._current_waypoint}/{len(self.trajectory)} reached")"""
        
        #total_reward = dist_reward + progress_reward
        total_reward = np.round((trajectory_reward + target_reward + step_reward),3)
        return total_reward

    def _get_obs(self):
        """Calcualte the observation index based on the current x,y,z position of the agent."""
        pos     = self._agent_location
        int_pos = np.round((pos - self.lower_bound) / self.resolution).astype(int)
        x_len   = int(np.round((self.upper_bound[0] - self.lower_bound[0]) / self.resolution))
        y_len   = int(np.round((self.upper_bound[1] - self.lower_bound[1]) / self.resolution))
        obs     = int_pos[0] + int_pos[1]*(x_len+1) + int_pos[2]*(x_len+1)*(y_len+1)  # +1 to include the last value
        return obs

    def _get_info(self):
        return {
            "tcp_position":         self._agent_location,
            "tcp_orientation":      self._tcp_pose(self._agent_angles)[3:],
            "target_distance":      self._distance('target'), 
            "trajectory_distance":  self._distance('trajectory'),
        }
    
    def _transform(self, joint_angles):
        """Calculate the individual transformation matrix for each joint 
        using the Denavit-Hartenberg matrix.
        """
        # Calculate the transformation matrix for each joint
        T = np.zeros((6,4,4))
        for n in range(6):
            # Extract the joint angles, alpha, a and d from the Denavit-Hartenberg matrix
            tetha = joint_angles[n]
            alpha = self.dh_matrix[n,2]
            a = self.dh_matrix[n,0]
            d = self.dh_matrix[n,1]

            # Convert the angles to radians
            alpha = np.deg2rad(alpha)
            tetha = np.deg2rad(tetha)

            T[n, 0, 0] = np.cos(tetha)
            T[n, 0, 1] = -1 * np.sin(tetha) * np.cos(alpha)
            T[n, 0, 2] = np.sin(tetha) * np.sin(alpha)
            T[n, 0, 3] = np.cos(tetha) * a

            T[n, 1, 0] = np.sin(tetha)
            T[n, 1, 1] = np.cos(tetha) * np.cos(alpha)
            T[n, 1, 2] = -1 * np.cos(tetha) * np.sin(alpha)
            T[n, 1, 3] = np.sin(tetha) * a

            T[n, 2, 0] = 0
            T[n, 2, 1] = np.sin(alpha)
            T[n, 2, 2] = np.cos(alpha)
            T[n, 2, 3] = d

            T[n, 3, 0] = 0
            T[n, 3, 1] = 0
            T[n, 3, 2] = 0
            T[n, 3, 3] = 1
        
        return T
    
    def _forward__kinematics(self, joint_angles):
        """Calculate the forward kinematics for each joint. 
        Returns the transformation matrix for each joint in relation to the origin.
        """
        T = self._transform(joint_angles)
        # Calculate the joint poses
        joints = np.zeros((7,4,4))
        for n in range(6):
            if n == 0:
                joints[n+1] = T[n]
            else:
                joints[n+1] = joints[n] @ T[n]
        return joints

    def _tcp_pose(self, joint_angles):
        """Calculate the pose of tool center point of the robot arm using the Denavit-Hartenberg matrix.
        Returns the x,y,z, alpha, beta, gamma position of the end effector.
        """
        # Iterate over all 6 axis and calculate the transformation matrix and corresponding joint poses
        joints = self._forward__kinematics(joint_angles)

        # extract the x,y,z position and the alpha, beta, gamma orientation from the joint poses
        # round to 3 decimal places 
        x = joints[6, 0, 3]
        y = joints[6, 1, 3]
        z = joints[6, 2, 3]
        
        # TODO check if this is correct (alpha, beta, gamma orientation)
        beta    = np.arctan2(-joints[6,2,0],np.sqrt(joints[6,0,0]**2 + joints[6,1,0]**2))
        alpha   = np.arctan2(joints[6,1,0]/np.cos(beta),joints[6,0,0]/np.cos(beta))
        gamma   = np.arctan2(joints[6,2,1]/np.cos(beta),joints[6,2,2]/np.cos(beta))

        alpha,beta,gamma = np.rad2deg([alpha,beta,gamma])

        # round to 3 decimal places
        x,y,z = np.round([x,y,z],3)
        alpha,beta,gamma = np.round([alpha,beta,gamma],1)

        # Return the x,y,z position and the alpha, beta, gamma orientation
        tcp_pose = np.array([x,y,z,alpha,beta,gamma])

       
        #print(tcp_pose)
        return tcp_pose

    def _render_frame(self):

        if self.fig is None and self.render_mode == "human":
            # Initialize the figure and clock
            plt.ion()
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(121, projection='3d')
            self.ax2 = self.fig.add_subplot(122, projection='3d')

        # Render the frame if the render mode is set to human and the current step is a multiple of the render skip
        if self.render_mode == "human" and self.num_steps % self.render_skip == 0:
            for n, self.ax in enumerate([self.ax1,self.ax2]):
                # Clear the plot
                self.ax.clear()

                # Set the axes limits
                if n == 0:
                    x_l = [-0.35 , 0.25]
                    y_l = [-0.35 , 0.35]
                    z_l = [0    , 0.5]
                elif n == 1:
                    x_l = [self.lower_bound[0], self.upper_bound[0]]
                    y_l = [self.lower_bound[1], self.upper_bound[1]]
                    z_l = [self.lower_bound[2], self.upper_bound[2]]

                self.ax.set(xlim3d=x_l, xlabel='X')
                self.ax.set(ylim3d=y_l, ylabel='Y')
                self.ax.set(zlim3d=z_l, zlabel='Z')

                # Extract the x,y,z position from the joint poses
                joint_poses = self._forward__kinematics(self._agent_angles)
                x = np.zeros(shape=(8))
                y = np.zeros(shape=(8))
                z = np.zeros(shape=(8))
                
                for n in range(7):
                    x[n+1] = joint_poses[n, 0, 3]
                    y[n+1] = joint_poses[n, 1, 3]
                    z[n+1] = joint_poses[n, 2, 3]

                # Draw the trajetory as a gray dotted line 
                t_x = [self.start[0], self.stop[0]]
                t_y = [self.start[1], self.stop[1]]
                t_z = [self.start[2], self.stop[2]]   
                self.ax.plot(t_x,t_y,t_z,label='trajectory', color = 'gray', linewidth = 1, linestyle = 'dotted')
                
                # Draw the observation space
                # Calculate edge lengths based on corner points
                start           = self.lower_bound
                size            = self.upper_bound - self.lower_bound

                # Define all 8 vertices of the cube based on corner points and edge lengths
                vertices = np.array([
                    start,
                    [start[0] + size[0], start[1], start[2]],
                    [start[0] + size[0], start[1] + size[1], start[2]],
                    [start[0], start[1] + size[1], start[2]],
                    [start[0], start[1], start[2] + size[2]],
                    [start[0] + size[0], start[1], start[2] + size[2]],
                    [start[0] + size[0], start[1] + size[1], start[2] + size[2]],
                    [start[0], start[1] + size[1], start[2] + size[2]],
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
                self.ax.plot(*edge.T[0], color='red', linewidth=1, linestyle = 'dotted', label='observation space', alpha=0.5)

                # Draw the robot arm
                self.ax.plot(x, y, z, markerfacecolor='k', markeredgecolor='b', marker='o', markersize=5, alpha=0.7, linewidth = 4)

                # Draw the tool center point (euler angle representation of the end effector orientation)
                tcp_x = np.zeros(shape=(4))
                tcp_y = np.zeros(shape=(4))
                tcp_z = np.zeros(shape=(4))

                scale = 0.05    # scale the length of the vectors
                m = 6           # joint 6 is the end effector

                for n in range(4):
                    tcp_x[n] = joint_poses[m, 0, n]*scale
                    tcp_y[n] = joint_poses[m, 1, n]*scale
                    tcp_z[n] = joint_poses[m, 2, n]*scale
                
                # iterate through the tcp points and colors
                for n,c in zip(range(3),['r','g','b']):
                    self.ax.plot([x[m+1],tcp_x[n]+x[m+1]], [y[m+1],tcp_y[n]+y[m+1]], [z[m+1],tcp_z[n]+z[m+1]], color = c, markersize=5, alpha=0.7, linewidth = 1)
                
                m = 1           # joint 6 is the end effector

                for n in range(4):
                    tcp_x[n] = joint_poses[m, 0, n]*scale
                    tcp_y[n] = joint_poses[m, 1, n]*scale
                    tcp_z[n] = joint_poses[m, 2, n]*scale
                
                # iterate through the tcp points and colors
                for n,c in zip(range(3),['r','g','b']):
                    self.ax.plot([x[m+1],tcp_x[n]+x[m+1]], [y[m+1],tcp_y[n]+y[m+1]], [z[m+1],tcp_z[n]+z[m+1]], color = c, markersize=5, alpha=0.7, linewidth = 1)
                
                # Draw plot and flush events to enable interactive mode
                self.ax.legend()
                self.fig.canvas.draw() # draw new contents
                self.fig.canvas.flush_events() 

    def close(self):
        # Close the figure
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    env = RobotArmEnv(render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        obs, reward, term, trunc, info = env.step(action)
        print(obs, reward, info)

    env.close()