import rospy

def vio_node():
    rospy.init_node('vio_node')

def integrate_state():
    '''
    Updates state incrementally every time IMU data is received
    <subscriber to imu topic>
    '''
    pass

def update_state():
    '''
    <subscriber to VO topic>
    Calls kalamn filter to update state using currently accrued IMU state and recently received VO
    Publishes /rtabmap/odom so the solver gets an odometry updatry
    '''
    pass

if __name__ == '__main__':
    try:
        vio_node()
    except rospy.ROSInterruptException:
        pass