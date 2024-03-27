import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.Kp = np.diag([15, 15, 23]) # 15, 23
        
        self.Kd = np.diag([5, 5, 8]) # 5, 8
        
        self.Kr = np.diag([144, 144, 18]) # 144, 18

        self.Kw = np.diag([13, 13, 7]) # 13 7
        
        self.gamma = self.k_drag / self.k_thrust # calculate gamma

        # define the A matrix for solving F1, F2, F3, F4 
        self.A = np.array([
            [1, 1, 1, 1],
            [0, self.arm_length, 0, -self.arm_length],
            [-self.arm_length, 0, self.arm_length, 0],
            [self.gamma, -self.gamma, self.gamma, -self.gamma]
        ])

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        # define the error for feedback control
        pos_rror =  state['x'] - flat_output['x'] 
        vel_error = state['v'] - flat_output['x_dot'] 
        r_ddot_des = flat_output['x_ddot'] - self.Kd @ vel_error - self.Kp @ pos_rror

        F_des = self.mass * (r_ddot_des + np.array([0, 0, self.g]))
        R = Rotation.from_quat(state['q']).as_matrix() # transform quaternion to rotation matrix
        b3 = R @ np.array([0, 0, 1])
        u1 = b3 @ F_des # calculate thrust

        b_3_des = F_des / np.linalg.norm(F_des) # the orientation of b3 is aligned with F_des
        psi_des = flat_output['yaw']
        a_psi = np.array([np.cos(psi_des), np.sin(psi_des), 0])
        b_2_des = np.cross(b_3_des, a_psi) / np.linalg.norm(np.cross(b_3_des, a_psi))
        b_1_des = np.cross(b_2_des, b_3_des)
        R_des = np.hstack((b_1_des.reshape(3,1), b_2_des.reshape(3,1), b_3_des.reshape(3,1))) # update the rotation matrix
        
        flag = R_des.T @ R - R.T @ R_des
        e_R = 0.5 * np.array([flag[2, 1], flag[0, 2], flag[1, 0]])
        # calculate the moment
        u2 = self.inertia @ (-self.Kr @ e_R - self.Kw @ state['w'])

        cmd_thrust = u1
        cmd_moment = u2
        cmd_q = Rotation.from_matrix(R_des).as_quat() # transform updated rotaion matrix to quaternion

        b = np.append(u1, u2)
        cmd_F = np.linalg.solve(self.A, b) # solve for F1, F2, F3, F4
        cmd_F = np.maximum(cmd_F, 0) # make sure all F are not negative
        cmd_motor_speeds = np.sqrt(cmd_F / self.k_thrust) # calculate the motor speed

        for i in range(4): # limit the motor speed within the range
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max
            elif cmd_motor_speeds[i] < self.rotor_speed_min:
                cmd_motor_speeds[i] = self.rotor_speed_min

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
