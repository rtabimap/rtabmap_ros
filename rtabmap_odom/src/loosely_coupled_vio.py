#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tf
from tf.msg import tfMessage
from tf.transformations import quaternion_matrix

from filterpy.kalman import KalmanFilter
import numpy as np

from scipy.spatial.transform import Rotation

class LooselyCoupledKF:
    #** Outline of what this node does**#
    # This node will take in two types of measurements / estimates
    # a) the imu data
    # b) the stereo odometry estimate from VO
    # 
    # We will use the IMU data in place of the motion model to predict where we have moved at a high frequency
    # At a lower frequency, the VO odometry estimate will be used to correct / update our prediction
    # In other words, a normal Kalman Filter.

    def __init__(self) -> None:
        delta_t = 0.05 # TODO hardcoded, update with message timestamps in subscriber, and all corresponding places where this is getting used

        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.integrate_state)
        self.vo_sub = rospy.Subscriber('/vo', Odometry, self.update_state)
        self.vio_publisher = rospy.Publisher('/odometry/filtered', Odometry, queue_size=10)
        self.tf_listener = tf.TransformListener()

        self.filter = KalmanFilter(12, 12, 6)

        # state defined as : x,y,z,phi,theta,psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot
        self.filter.x = np.zeros((12,))
        rospy.loginfo("Created state of size {}".format(self.filter.x.shape[0]))

        # State transition matrix (one that takes previous state to this one in this timestep)
        F = np.eye(12)
        F[0,6] = delta_t
        F[1, 7] = delta_t
        F[2, 8] = delta_t
        F[6, 9] = delta_t
        F[7, 10] = delta_t
        F[8, 11] = delta_t
        F[9, :] = 0
        F[10, :] = 0
        F[11, :] = 0
        self.filter.F = F

        # current state covariance matrix (keeps getting updated with state estimate)
        self.filter.P = np.eye(12) #TODO can update the initialization here

        # Measurement noise matrix
        self.filter.R = np.eye(6) #TODO some tuning here

        # process noise matrix
        self.filter.Q = np.eye(12)

        # Measurement function (gets the state from odometry measurements)
        H = np.eye(12)
        self.filter.H = H

        # Control transition matrix (gets state update from imu input)
        B = np.zeros((12, 6))
        B[6, 0] = delta_t
        B[7, 1] = delta_t
        B[8, 2] = delta_t
        B[9, 3] = 1
        B[10, 3] = 1
        B[11, 3] = 1
        self.filter.B = B

        # TODO UPDATE THIS WITH ACTUAL TRANSFORM!!!!!!!!!!!!!!
        # imu to VO tf
        self.imu_tf = np.eye(3)

        rospy.loginfo("Starting KF")


    def update_imu_tf(self, tf_message):
    
        if (tf_message.child_frame_id == "imu4"):
            
            # collect the rotation quaternion
            quat = tf_message.transforms[0].transform.rotation

            # Build an array with the quaternion values
            quat_array = [0,0,0,0]
            quat_array[0] = quat.x
            quat_array[1] = quat.y
            quat_array[2] = quat.z
            quat_array[3] = quat.w

            # Update imu_tf with the new quaternion
            imu_tf = quaternion_matrix(quat)


    def integrate_state(self, imu_msg):
        '''
        Input:

            imu_msg: Imu type sensor msg
        Updates state incrementally every time IMU data is received
        <subscriber to imu topic>
        '''
        x_ddot = imu_msg.linear_acceleration.x
        y_ddot = imu_msg.linear_acceleration.y
        z_ddot = imu_msg.linear_acceleration.z
        theta_dot = imu_msg.angular_velocity.x
        phi_dot = imu_msg.angular_velocity.y
        psi_dot = imu_msg.angular_velocity.z
        ang_vel_cov = imu_msg.angular_velocity_covariance
        ang_vel_cov = np.array(ang_vel_cov).reshape(3,3)
        lin_acc_cov = imu_msg.linear_acceleration_covariance
        lin_acc_cov = np.array(lin_acc_cov).reshape(3,3)

        curr_state = self.filter.x
        curr_theta = curr_state[3]
        curr_phi = curr_state[4]
        curr_psi = curr_state[5]

        # transformation of angular velocities from IMU frame to world frame https://www.mathworks.com/help/aeroblks/6dofeulerangles.html

        ang_vel_tf = np.eye(3)
        
        ang_vel_tf[0,1] = np.sin(curr_phi) * np.tan(curr_theta)
        ang_vel_tf[0,2] = np.cos(curr_phi) * np.tan(curr_theta)
        ang_vel_tf[1,1] = np.cos(curr_phi)
        ang_vel_tf[1,2] = -np.sin(curr_phi)
        ang_vel_tf[2,1] = np.sin(curr_phi)/np.cos(curr_theta)
        ang_vel_tf[2,2] = np.cos(curr_phi)/np.cos(curr_theta)

        # getting world frame ang vels
        w_ang_vel = ang_vel_tf @ np.array([theta_dot, phi_dot, psi_dot]).reshape(-1,1)

        # getting world frame accns
        w_accns = self.imu_tf @ np.array([x_ddot, y_ddot, z_ddot]).reshape(-1,1) + np.array([0,0,9.8]).reshape(-1,1)
        imu_input = np.concatenate([w_accns, w_ang_vel]).squeeze(-1)
        # update filterpy.Q (process noise matrix)
        Q = np.eye(12) #TODO may want to update other elements of measurement covariances too
        # TODO: not using the imu acceleration covariance data, how can we incorporate it?
        Q[-3:, -3:] = ang_vel_cov

        self.filter.predict(u=imu_input, Q=Q)

    def update_state(self, odom_msg):
        '''
        Input:
            odom_msg: Odometry nav_msg
        <subscriber to VO topic>
        Calls kalamn filter to update state using currently accrued IMU state and recently received VO
        Publishes /rtabmap/odom so the solver gets an odometry updatry
        '''
        # update filterpy.R (measurement noise matrix) using odometry message (pose with covariance & twist with covariance)

        # extract data from odom_msg, create measurement z(quaternion to euler) and call the update function

        # after filtering, convert back state into quaternion and publish
        odom_x = odom_msg.pose.pose.position.x
        odom_y = odom_msg.pose.pose.position.y
        odom_z = odom_msg.pose.pose.position.z
        odom_quat_x = odom_msg.pose.pose.orientation.x
        odom_quat_y= odom_msg.pose.pose.orientation.y
        odom_quat_z = odom_msg.pose.pose.orientation.z
        odom_quat_w = odom_msg.pose.pose.orientation.w
        odom_pose_cov = np.array([odom_msg.pose.covariance]).reshape(6,6)
        odom_x_dot = odom_msg.twist.twist.linear.x
        odom_y_dot = odom_msg.twist.twist.linear.y
        odom_z_dot = odom_msg.twist.twist.linear.z
        odom_twist_theta = odom_msg.twist.twist.angular.x
        odom_twist_phi = odom_msg.twist.twist.angular.y
        odom_twist_psi = odom_msg.twist.twist.angular.z
        odom_twist_cov = np.array([odom_msg.twist.covariance]).reshape(6,6)

        if (np.linalg.norm([odom_quat_x, odom_quat_y, odom_quat_z, odom_quat_w]) < 0.8):
            rospy.logwarn("Bad VO quaternion received. Skipping update step.")
            return
        rot = Rotation.from_quat([odom_quat_x, odom_quat_y, odom_quat_z, odom_quat_w])
        odom_euler = rot.as_euler('xyz')

        # input
        z = np.zeros((12,))

        z[0] = odom_x
        z[1] = odom_y
        z[2] = odom_z
        z[3] = odom_euler[0]
        z[4] = odom_euler[1]
        z[5] = odom_euler[2]
        z[6] = odom_x_dot
        z[7] = odom_y_dot
        z[8] = odom_z_dot
        z[9] = odom_twist_theta
        z[10] = odom_twist_phi
        z[11] = odom_twist_psi

        # measurement noise matrix

        R = np.eye(12)
        R[:6, :6] = odom_pose_cov
        R[6:,6:] = odom_twist_cov

        self.filter.update(z= z, R=R)

        new_state = self.filter.x
        new_cov = self.filter.P

        rot = Rotation.from_euler('xyz', [new_state[3], new_state[4], new_state[5]])
        quaternion = rot.as_quat()

        filtered_state = Odometry()
        filtered_state.pose.pose.position.x = new_state[0]
        filtered_state.pose.pose.position.y = new_state[1]
        filtered_state.pose.pose.position.z = new_state[2]
        filtered_state.pose.pose.orientation.x = quaternion[0]
        filtered_state.pose.pose.orientation.y = quaternion[1]
        filtered_state.pose.pose.orientation.z = quaternion[2]
        filtered_state.pose.pose.orientation.w = quaternion[3]
        filtered_state.pose.covariance = new_cov[:6,:6].flatten()
        filtered_state.twist.twist.linear.x = new_state[6]
        filtered_state.twist.twist.linear.y = new_state[7]
        filtered_state.twist.twist.linear.z = new_state[8]
        filtered_state.twist.twist.angular.x = new_state[9]
        filtered_state.twist.twist.angular.y = new_state[10]
        filtered_state.twist.twist.angular.z = new_state[11]
        filtered_state.twist.covariance = new_cov[6:,6:].flatten()

        # TODO: add appropriate header with timestamps and frame ids
        filtered_state.header = odom_msg.header
        filtered_state.child_frame_id = odom_msg.child_frame_id

        self.vio_publisher.publish(filtered_state)


if __name__ == '__main__':

    rospy.init_node('vio_node')
    try:
        kf = LooselyCoupledKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
