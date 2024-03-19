#!/bin/python3
# Python 2/3 compatibility imports
import matplotlib.pyplot as plt
import numpy
import numpy as np
import rospy,sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry msgs.msg import Pose
from std_msgs.msg import String
from copy import deepcopy
'''
DeltaPID类用于存放用于PID控制的代码,其类型为增量PID控制。
PID控制器的输入量为三个向量，输出量为两个向量：
输入量：
1、胶囊在xy平面上的实际位置与目标位置的偏差(epxy)（2*1）
2、胶囊在z轴方向上的实际位置与目标位置的偏差(fz)（1*1）
3、胶囊实际航向与目标航向之间偏差(eh)(3*1)
输出量：
1、胶囊每次迭代受力的改变量δfmd
2、胶囊每次迭代磁矩的改变量δτmd
'''
class DeltaPID(object):
    """增量式PID算法实现"""

    def __init__(self, target, cur_val, dt, p, i, d) -> None:
        self.dt = dt  # 循环时间间隔
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数

        self.target = target  # 目标值
        self.cur_val = cur_val  # 算法当前PID位置值
        self._pre_error = 0  # t-1 时刻误差值
        self._pre_pre_error = 0  # t-2 时刻误差值

    def calcalate(self):
        error = self.target - self.cur_val
        p_change = self.k_p * (error - self._pre_error)
        i_change = self.k_i * error
        d_change = self.k_d * (error - 2 * self._pre_error + self._pre_pre_error)
        delta_output = p_change + i_change + d_change  # 本次增量
        self.cur_val += delta_output  # 计算当前位置

        self._pre_pre_error = self._pre_error

        self._pre_error = error

        return self.cur_val


    def fit_and_plot(self, count=200):
        counts = np.arange(count)
        outputs = []
        for i in counts:
            outputs.append(self.calcalate())
            print('Count %3d: output: %f' % (i, outputs[-1]))

        print('Done')

        plt.figure()
        plt.axhline(self.target, c='red')
        plt.plot(counts, np.array(outputs), 'b.')
        plt.ylim(min(outputs) - 0.1 * min(outputs),
                max(outputs) + 0.1 * max(outputs))
        plt.plot(outputs)
        plt.show()
'''
Sensor类用于存放获取数据的ROS通信代码，包括四种参数：
1、胶囊的位置向量Pe（3*1）
2、胶囊的方向向量Me（3*1）
3、末端执行器磁铁的位置向量Pa（3*1）
4、末端执行器磁铁的方向向量Ma（3*1）
'''
class Sensor:
    #获取胶囊航向的传感器数据
    def callback_Me(self,data):
        rospy.loginfo(rospy.get_caller_id(),data.data)
        global Me_hat
        Me_hat=data.data

    def listener_Me(self):
        rospy.init_node('listener_Me',anonymous=True)
        rospy.Subscriber('chatter_position_sensor',String,callback_Me)
        rospy.spin()
'''
Navigation类用于存放实现项目的主要代码，其编写思路如下：
1、初始化API,ROS节点等
2、读取各类输入量，包括机械臂当前的雅可比矩阵、胶囊与磁铁的位姿向量等
*3、设置胶囊导航的目标位置坐标
4、进行PID计算，输入量为3个偏差，输出量为2个理想受力
5、计算雅可比矩阵Je，并以此计算胶囊位置改变量δPe与胶囊航向改变量δMe
6、计算雅可比矩阵Ja并求出其伪逆矩阵，得到最终目标参量δq（6*1）

'''
class Navigation:
    def __init__(self):

        '''
        1、初始化API,ROS节点等
        '''
        #初始化API
        moveit_commander.roscpp_initialize(sys.argv)
        #初始化ROS节点
        rospy.init_node('Navigation',anonymous=True)
        #初始化move group控制的机械臂中的arm group
        arm=MoveGroupCommander('manipulator')
        #设置关节最大误差
        arm.set_goal_joint_tolerance(0.001)
        #设置速度加速度缩放比例
        arm.set_max_acceleration_scaling_factor(0.5)
        arm.set_max_velocity_scaling_factor(0.5)
        '''
        2、读取各类输入量，包括机械臂当前的雅可比矩阵、胶囊与磁铁的位姿向量等
        '''
        #获取当前的机械臂各关节值
        current_joint_values=arm.get_current_joint_values()
        #获取当前机械臂的雅可比矩阵Jr
        J_r=numpy.zeros((6,6))
        J_r=arm.get_jacobian_matrix(current_joint_values)

        # 获取传感器传回的胶囊位姿数据，待修改
        Sensor.listener_Me()
        Sensor.callback_Me(data)












        '''
        6、计算雅可比矩阵Ja并求出其伪逆矩阵，得到最终目标参量δq（6*1）
        '''
        #求出机械臂末端磁铁的雅可比矩阵Ja
        unit_matrix=numpy.eye(3)
        zeros_matrix=numpy.zeros((3,3))
        #计算雅可比矩阵Ja
        S_matrix=numpy.matrix([[0,-1*Me_hat(2),Me_hat(1)],[Me_hat(2),0,-1*Me_hat(0)],[-1*Me_hat(1),Me_hat(0),0]])
        S_matrix_t=numpy.transpose(S_matrix)
        J_a_up=numpy.hstack((unit_matrix,zeros_matrix))
        J_a_down=numpy.hstack((zeros_matrix,S_matrix_t))
        J_a=numpy.vstack((J_a_up,J_a_down))

