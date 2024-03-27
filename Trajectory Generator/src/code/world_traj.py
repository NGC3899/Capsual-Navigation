import numpy as np

from .graph_search import graph_search

import numpy as np

from scipy import optimize

class Build_constrain_matrix(object):
    @staticmethod
    def build_H(T):
        H = np.array([
            [720 * T**5, 360 * T**4, 120 * T**3, 0, 0, 0],
            [360 * T**4, 192 * T**3, 72 * T**2, 0, 0, 0],
            [120 * T**3, 72 * T**2, 36 * T, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
            ])
        return H

    @staticmethod
    def build_big_H(H_block):
        
        block_matrix = np.array(H_block[0])

        for matrix in H_block[1:]:
            block_matrix = np.block([
                [block_matrix, np.zeros((block_matrix.shape[0], len(matrix)))],
                [np.zeros((len(matrix), block_matrix.shape[1])), matrix]
            ])
        return block_matrix

    @staticmethod
    def conti_block(T):
        conti_block = np.array([
        [T**5, T**4, T**3, T**2, T, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0, 0, 0, 0, 0, -1, 0],
        [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0, 0, 0, 0, -2, 0, 0],
        [60 * T**2, 24 * T, 6, 0, 0, 0, 0, 0, -6, 0, 0, 0],
        [120 * T, 24, 0, 0, 0, 0, 0, -24, 0, 0, 0, 0]
        ])
        return conti_block

    @staticmethod
    def bulid_constrain_block(time, N):
        """
        time: time of each segement, (N-1,)
        N: waypoints number
        """
        row = 6 + 6 * (N - 2)
        column = 6 * (N-1)
        A = np.zeros((row, column))
        # start constrain
        A[0, 5] = 1
        A[1, 4] = 1
        A[2, 3] = 2

        # end constrain
        T_e = time[-1]
        
        A[3:6, -6:] = np.array([
            [T_e**5, T_e**4, T_e**3, T_e**2, T_e, 1],
            [5 * T_e**4, 4 * T_e**3, 3 * T_e**2, 2 * T_e, 1, 0],
            [20 * T_e**3, 12 * T_e**2, 6 * T_e, 2, 0, 0]
            ])

        for i in range(N-2):
            A[6*(i+1):6*(i+1) + 6, 6*i:6*i + 12] = Build_constrain_matrix.conti_block(time[i])

        return A

    @staticmethod
    def bulid_b(points, N):
        """
        points: x/y/z coordinates of waypoints
        """
        start = points[0]
        end = points[-1]
        b = np.array([start, 0, 0, end, 0, 0])
        for i in range(N-2):
            b_update = np.array([points[i+1], points[i+1], 0, 0, 0, 0])
            b = np.concatenate((b, b_update))
        return b

class Simplify_points(object):
    @staticmethod
    def RDP(position, insert, start_idx, end_idx, epsilon):

        if start_idx >= end_idx - 1:
            return

        max_dist = 0.0
        line_vec = position[end_idx] - position[start_idx]
        id = -1

        for i in range(start_idx+1, end_idx):
            point_vec = position[i] - position[start_idx]
            v = np.dot(point_vec, line_vec) / (np.linalg.norm(point_vec) * np.linalg.norm(line_vec))
            if -1 <= v <= 1:
                theta = abs(np.arccos(v))
            else:
                continue
            dist = abs(np.linalg.norm(point_vec) * np.sin(theta))

            if max_dist <= dist:
                max_dist = dist
                id = i

        if max_dist <= epsilon:
            return

        insert.append(id)

        Simplify_points.RDP(position, insert, id, end_idx, epsilon)
        Simplify_points.RDP(position, insert, start_idx, id, epsilon)
        return

    @staticmethod
    def simplified(position, threshold):
        insert = []
        start_idx = 0
        end_idx = len(position) - 1
        Simplify_points.RDP(position, insert, start_idx, end_idx, threshold)

        if len(position) == 0:
            print("Graph search not found")
            return None

        if len(insert) == 0 and len(position) != 0:
            print("RDP error")
            return None

        insert = np.asarray(insert)
        sorted_path = np.sort(insert)

        if sorted_path[0] != 0:
            sorted_path = np.insert(sorted_path, 0, 0)
        if sorted_path[len(sorted_path)-1] != len(position)-1:
            sorted_path = np.append(sorted_path, len(position)-1)

        l = 0
        h = 1
        s = len(sorted_path)
        while h < s:
            if sorted_path[h] - sorted_path[l] > 14:
                sorted_path = np.insert(sorted_path, h, (sorted_path[h] + sorted_path[l]) // 2)
                s += 1
            else:
                l += 1
                h += 1

        simplified_position = np.zeros((len(sorted_path), 3))
        for i in range(0, len(sorted_path)):
            simplified_position[i] = position[sorted_path[i]]

        return simplified_position


class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.15, 0.15, 0.15])
        self.margin = 0.1
        self.v = 2.8 # not fixed velocity
        epsilon = 0.5 # bigger meams less points

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True) # store the list of path points

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        
        
        # rdp_algorithm = RDPAlgorithm(epsilon)
        # simplified_points = rdp_algorithm.simplify(self.path) # simplified the path data with Ramer–Douglas–Peucker algorithm
        simplified_points = Simplify_points.simplified(self.path, epsilon)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        
        self.points = np.array(simplified_points)
        N, _ = self.points.shape # the N of points

        time = []

        # calculate the time period T
        for i in range(N-1):
            seg = self.points[i+1, :] - self.points[i, :] # the vector between two points
            seg_norm = np.linalg.norm(seg) # the distance between two points
       
            time.append(seg_norm / self.v)

        time[0] = 1.4 * time[0]   
        time[-1] = 1.4 * time[-1]

        self.time = np.array(time) # (N-1, ) matrix, time of a segment

        self.time_start = np.zeros((N, ))
        for i in range(1, N):
            self.time_start[i] = self.time_start[i-1] + self.time[i-1]

        H_block = []
        for i in range(N-1):
            H = Build_constrain_matrix.build_H(self.time[i])
            H_block.append(H)
        
        Big_H = Build_constrain_matrix.build_big_H(H_block)
            
        A = Build_constrain_matrix.bulid_constrain_block(self.time, N)

        points_x = self.points[:, 0]
        b_x = Build_constrain_matrix.bulid_b(points_x, N)
        
        points_y = self.points[:, 1]
        b_y = Build_constrain_matrix.bulid_b(points_y, N)

        points_z = self.points[:, 2]
        b_z = Build_constrain_matrix.bulid_b(points_z, N)       
        
        def loss(x):
            return 0.5 * np.dot(x.T, np.dot(Big_H, x))

        def solve(A, b):
            cons = {'type':'eq', 'fun': lambda x: np.dot(A, x) - b}
            x0 = np.random.rand(A.shape[1])

            res_cons = optimize.minimize(loss, x0, constraints = cons, method='SLSQP')
            return res_cons
        
        self.coff_x = solve(A, b_x).x
        self.coff_y = solve(A, b_y).x
        self.coff_z = solve(A, b_z).x

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        flag = -1

        # STUDENT CODE HERE
        if t < self.time_start[-1]: # quadrotor doesn't reach terminal
            for i in self.time_start: # judge which segment the quadrotor is in
                if t >= i:
                    flag += 1
                else:
                    break
            coff_x = self.coff_x[flag * 6 : flag * 6 + 6]
            coff_y = self.coff_y[flag * 6 : flag * 6 + 6]
            coff_z = self.coff_z[flag * 6 : flag * 6 + 6]
            spline_x = np.polyval(coff_x, t - self.time_start[flag])
            spline_y = np.polyval(coff_y, t - self.time_start[flag])
            spline_z = np.polyval(coff_z, t - self.time_start[flag])
            x = np.array([spline_x, spline_y, spline_z]) # calculate the position of quadrotor
            
            dot_coff = np.array([5, 4, 3, 2, 1])
            coff_x_dot = dot_coff * coff_x[:-1]
            coff_y_dot = dot_coff * coff_y[:-1]
            coff_z_dot = dot_coff * coff_z[:-1]
            spline_x_dot = np.polyval(coff_x_dot, t - self.time_start[flag])
            spline_y_dot = np.polyval(coff_y_dot, t - self.time_start[flag])
            spline_z_dot = np.polyval(coff_z_dot, t - self.time_start[flag])
            x_dot = np.array([spline_x_dot, spline_y_dot, spline_z_dot])

            ddot_coff = np.array([20, 12, 6, 2])
            coff_x_ddot = ddot_coff * coff_x[:-2]
            coff_y_ddot = ddot_coff * coff_y[:-2]
            coff_z_ddot = ddot_coff * coff_z[:-2]
            spline_x_ddot = np.polyval(coff_x_ddot, t - self.time_start[flag])
            spline_y_ddot = np.polyval(coff_y_ddot, t - self.time_start[flag])
            spline_z_ddot = np.polyval(coff_z_ddot, t - self.time_start[flag])
            x_ddot = np.array([spline_x_ddot, spline_y_ddot, spline_z_ddot])
            
        else: # if quadrotor arrive at the terminal, the velocity is zero
            x = self.points[-1, :] 
            x_dot = np.zeros((3, ))
            x_ddot = np.zeros((3, ))

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
