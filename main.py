#!/bin/python3
# Python 2/3 compatibility imports
import numpy
import rospy,sys
import moveit_commander
from moveit_commander import MoveGroupCommander
from geometry msgs.msg import Pose
from std_msgs.msg import String
from copy import deepcopy


class Sensor:
    #获取胶囊航向的传感器数据
    def callback_Me(self,Me):
        rospy.loginfo(rospy.get_caller_id(),Me.data)
        Me_hat=Me.data
        return Me_hat
    def listener_Me(self):
        rospy.init_node('listener_Me',anonymous=True)
        rospy.Subscriber('chatter_position_sensor',String,callback)
        rospy.spin()
class Navigation:
    def __init__(self):
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
        #获取当前的机械臂各关节值
        current_joint_values=arm.get_current_joint_values()
        #获取当前机械臂的雅可比矩阵Jr
        J_r=numpy.zeros((6,6))
        J_r=arm.get_jacobian_matrix(current_joint_values)

        # 获取传感器传回的胶囊位姿数据，待修改
        Sensor.listener_Me()
        Sensor.callback_Me(Me)

        #求出机械臂末端磁铁的雅可比矩阵Ja
        unit_matrix=numpy.eye(3)
        zeros_matrix=numpy.zeros((3,3))

        S_matrix=numpy.matrix([[0,-1*Me_hat(2),Me_hat(1)],[Me_hat(2),0,-1*Me_hat(0)],[-1*Me_hat(1),Me_hat(0),0]])
        S_matrix_t=numpy.transpose(S_matrix)
        J_a_up=numpy.hstack((unit_matrix,zeros_matrix))
        J_a_down=numpy.hstack((zeros_matrix,S_matrix_t))
        J_a=numpy.vstack((J_a_up,J_a_down))

