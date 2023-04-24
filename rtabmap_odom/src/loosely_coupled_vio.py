#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tf
from tf.msg import tfMessage
from tf.transformations import quaternion_matrix

from filterpy.kalman import KalmanFilter
import numpy as np
from geometry_msgs.msg import TransformStamped
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
        delta_t = 0.005 # TODO hardcoded, update with message timestamps in subscriber, and all corresponding places where this is getting used

        # Create a transform broadcaster
        self.odom_tf_broadcaster = tf.TransformBroadcaster()

        # Create filtered odometry message
        self.filtered_state = Odometry()
        self.filtered_state.child_frame_id = "base_link"
        self.filtered_state.header.frame_id = "odom"
        self.filtered_state.header.seq = 0

        # self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.integrate_state)
        self.vo_sub = rospy.Subscriber('/vo', Odometry, self.update_state)
        self.vio_publisher = rospy.Publisher('/odometry/filtered', Odometry, queue_size=10)
        self.tf_listener = tf.TransformListener()

        self.filter = KalmanFilter(12, 12, 6)

        # state defined as : x,y,z,phi,theta,psi, x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot
        self.filter.x = np.zeros((12,))
        self.filter.x[:3] = [4.349, 1.805, -1.008]
        #self.initialise_with_ground_truth()
        rospy.loginfo("Created state of size {}".format(self.filter.x.shape[0]))

        # State transition matrix (one that takes previous state to this one in this timestep)
        F = np.eye(12)
        # F[0,6] = delta_t 
        # F[1, 7] = delta_t
        # F[2, 8] = delta_t
        # F[6, 9] = delta_t
        # F[7, 10] = delta_t
        # F[8, 11] = delta_t
        # # F[9, :] = 0
        # # F[10, :] = 0
        # # F[11, :] = 0
        self.filter.F = F

        # current state covariance matrix (keeps getting updated with state estimate)
        self.filter.P = np.eye(12) #TODO can update the initialization here

        # Measurement noise matrix
        self.filter.R = 1e-6 * np.eye(6) #TODO some tuning here

        # process noise matrix
        self.filter.Q = 1e6 * np.eye(12)

        # Measurement function (gets the state from odometry measurements)
        H = np.eye(12)
        self.filter.H = H

        # Control transition matrix (gets state update from imu input)
        B = np.zeros((12, 6))
        # B[6, 0] = delta_t
        # B[7, 1] = delta_t
        # B[8, 2] = delta_t
        # B[9, 3] = 1
        # B[10, 3] = 1
        # B[11, 3] = 1
        self.filter.B = B

        # TODO UPDATE THIS WITH ACTUAL TRANSFORM!!!!!!!!!!!!!!
        # imu to VO tf
        self.imu_tf = np.eye(4)
        self.update_imu_tf()

        print("IMU static tf: ", self.imu_tf)
        rospy.loginfo("Starting KF")

    def initialise_with_ground_truth(self):
        print("Updating initial state with ground truth!")

        try:
            now = rospy.Time.now()
            rospy.loginfo("Waiting for ground truth tf...")
            self.tf_listener.waitForTransform("/base_link_gt", "/world", now, rospy.Duration(10.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link_gt', '/world', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get ground truth. Skipping.")
            return

        self.filter.x[0] = trans[0]
        self.filter.x[1] = trans[1]
        self.filter.x[2] = trans[2]
        
        rot_euler = Rotation.from_quat(rot).as_euler('xyz') # TODO check roll,pitch, yaw mapping
        self.filter.x[3] = rot_euler[0]
        self.filter.x[4] = rot_euler[1]
        self.filter.x[5] = rot_euler[2]


    def update_imu_tf(self):
        try:
            now = rospy.Time.now()
            rospy.loginfo("Waiting for imu tf...")
            self.tf_listener.waitForTransform("/base_link", "/imu4", now, rospy.Duration(10.0))
            (trans, rot) = self.tf_listener.lookupTransform('/base_link', '/imu4', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get imu. Skipping.")
            return

        rot_mat = Rotation.from_quat(rot).as_matrix()
        self.imu_tf[:3, :3] = rot_mat
        self.imu_tf[:3, 3] = trans


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
        w_accns = self.imu_tf @ np.array([x_ddot, y_ddot, z_ddot, 1]).reshape(-1,1) + np.array([0,0,9.8,0]).reshape(-1,1)
        w_accns = w_accns[:-1]
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
        odom_quat_y = odom_msg.pose.pose.orientation.y
        odom_quat_z = odom_msg.pose.pose.orientation.z
        odom_quat_w = odom_msg.pose.pose.orientation.w
        odom_pose_cov = np.array([odom_msg.pose.covariance]).reshape(6,6)
        odom_x_dot = odom_msg.twist.twist.linear.x
        odom_y_dot = odom_msg.twist.twist.linear.y
        odom_z_dot = odom_msg.twist.twist.linear.z
        odom_twist_x = odom_msg.twist.twist.angular.x
        odom_twist_y = odom_msg.twist.twist.angular.y
        odom_twist_z = odom_msg.twist.twist.angular.z
        odom_twist_cov = np.array([odom_msg.twist.covariance]).reshape(6,6)

        if (np.linalg.norm([odom_quat_x, odom_quat_y, odom_quat_z, odom_quat_w]) < 0.8):
            rospy.logwarn("Bad VO quaternion received. Skipping update step.")
            return
        rot = Rotation.from_quat([odom_quat_x, odom_quat_y, odom_quat_z, odom_quat_w])
        odom_euler = rot.as_euler('xyz')

        # input
        z = np.zeros((12,))

        scale = 1

        z[0] = odom_x * scale
        z[1] = odom_y * scale
        z[2] = odom_z * scale
        z[3] = odom_euler[0]
        z[4] = odom_euler[1] # TODO: These odom_euler angeles need to be kept within the appropriate ranges
        z[5] = odom_euler[2]
        z[6] = odom_x_dot
        z[7] = odom_y_dot
        z[8] = odom_z_dot
        z[9] = odom_twist_x
        z[10] = odom_twist_y
        z[11] = odom_twist_z

        # measurement noise matrix

        R = 1e-6 * np.eye(12)
        # R[:6, :6] = odom_pose_cov # TODO: Can this be copied like this if the noise was in quaternion?
        # R[6:,6:] = odom_twist_cov 

        # print(R)
        # print()



        # return
        self.filter.update(z=z, R=R)

        # self.filter.x = z
        new_state = self.filter.x
        new_cov = self.filter.P

        # Publish odom
        # self.publish_odom(z,R)
        self.publish_odom(new_state,new_cov)    

        # Publish it to tf
        # self.publish_tf(z)
        self.publish_tf(new_state)
    
    def publish_odom(self,state, cov):

        rot = Rotation.from_euler('xyz', [state[3], state[4], state[5]])
        quaternion = rot.as_quat()

        self.filtered_state.header.seq += 1
        filtered_state = self.filtered_state
        filtered_state.pose.pose.position.x = state[0]
        filtered_state.pose.pose.position.y = state[1]
        filtered_state.pose.pose.position.z = state[2]
        filtered_state.pose.pose.orientation.x = quaternion[0]
        filtered_state.pose.pose.orientation.y = quaternion[1]
        filtered_state.pose.pose.orientation.z = quaternion[2]
        filtered_state.pose.pose.orientation.w = quaternion[3]
        filtered_state.pose.covariance = cov[:6,:6].flatten()
        filtered_state.twist.twist.linear.x = state[6]
        filtered_state.twist.twist.linear.y = state[7]
        filtered_state.twist.twist.linear.z = state[8]
        filtered_state.twist.twist.angular.x = state[9]
        filtered_state.twist.twist.angular.y = state[10]
        filtered_state.twist.twist.angular.z = state[11]
        filtered_state.twist.covariance = cov[6:,6:].flatten()

        # TODO: add appropriate header with timestamps and frame ids
        filtered_state.header.stamp = rospy.Time.now()

        self.vio_publisher.publish(filtered_state)

    def publish_tf(self, state):

        rot = Rotation.from_euler('xyz', [state[3], state[4], state[5]])
        quaternion = rot.as_quat()

        # Create a transform message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        # transform.header.frame_id = "base_link"
        # transform.child_frame_id = "odom"
        transform.transform.translation.x = state[0]
        transform.transform.translation.y = state[1]
        transform.transform.translation.z = state[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Publish the transform
        self.odom_tf_broadcaster.sendTransformMessage(transform)


if __name__ == '__main__':

    rospy.init_node('vio_node')
    try:
        kf = LooselyCoupledKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
