import numpy as np


class RobotArm:
    """
    Represents a X-DOF robotic arm with its configuration and functionalities.
    Where X is the number of joints specified by the Denavit-Hartenberg matrix.

    Args:
        dh_matrix (numpy.ndarray): 
            The Denavit-Hartenberg matrix for the robot arm.
        init (list): 
            The initial joint angles of the robot arm in degrees.
        dec (list): 
            The number of decimal places for the observation space and action space.

    Methods:
        forward_kinematics(self, joint_angles):
            Calculates the forward kinematics for each joint, returning a list of
            transformation matrices.
        get_tcp_pose(self, joint_angles):
            Calculates the Tool Center Point (TCP) pose of the robot arm for 
            given joint angles. Returns a NumPy array containing the TCP pose
            (x, y, z, alpha, beta, gamma).
    """

    def __init__(self, dh_matrix, init, dec):
        self.dh_matrix = dh_matrix
        self.init = init
        self.dec = dec
        self.resolution = [10**-self.dec[0], 10**-self.dec[1]]
        self.num_joints = self.dh_matrix.shape[0] 

    def _transform(self, joint_angles):
        """
        Calculates the individual transformation matrix for each joint.
        """
        # Calculate the transformation matrix for each joint
        T = np.zeros((self.num_joints,4,4))
        for n in range(self.num_joints):
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

    def get_fw_kin(self, joint_angles):
        """
        Calculates the forward kinematics for each joint, returning the transformation matrices.

        Args:
            joint_angles (list): A list of joint angles in degrees.

        Returns:
            numpy.ndarray: A list of 4x4 transformation matrices for each joint.
        """
        T = self._transform(joint_angles)
        joints = np.zeros((self.num_joints+1,4,4))
        for n in range(self.num_joints):
            if n == 0:
                joints[n+1] = T[n]
            else:
                joints[n+1] = joints[n] @ T[n]
        return joints

    def get_tcp_pose(self, joint_angles):
        """
        Calculates the Tool Center Point (TCP) pose of the robot arm.

        Args:
            joint_angles (list): A list of joint angles in degrees.

        Returns:
            numpy.ndarray: A NumPy array containing the TCP pose (x, y, z, alpha, beta, gamma).
        """
        # Iterate over all axis and calculate the transformation matrix and corresponding joint poses
        joints  = self.get_fw_kin(joint_angles)
        num     = self.num_joints

        # extract the x,y,z position and the alpha, beta, gamma orientation from the joint poses
        x = joints[num, 0, 3]
        y = joints[num, 1, 3]
        z = joints[num, 2, 3]
        
        # calculate the euler angles alpha, beta, gamma from the rotation matrix
        beta  = np.arctan2(-joints[num,2,0],np.sqrt(joints[num,0,0]**2 + joints[num,1,0]**2))
        alpha = np.arctan2(joints[num,1,0]/np.cos(beta),joints[num,0,0]/np.cos(beta))
        gamma = np.arctan2(joints[num,2,1]/np.cos(beta),joints[num,2,2]/np.cos(beta))

        # normalize angles
        alpha, beta, gamma = alpha % (2*np.pi), beta % (2*np.pi), gamma % (2*np.pi)

        # convert angles to degrees
        alpha,beta,gamma = np.rad2deg([alpha,beta,gamma])

        # round to decimal places
        x,y,z            = np.round([x,y,z],self.dec[0])
        alpha,beta,gamma = np.round([alpha,beta,gamma],self.dec[1])

        # Return the x,y,z position and the alpha, beta, gamma orientation
        tcp_pose = np.array([x,y,z,alpha,beta,gamma])

        return tcp_pose
