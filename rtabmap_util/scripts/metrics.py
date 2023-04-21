import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime
import tf
import rospy
import rospkg
from std_msgs.msg import Float32

class MetricsLogger:

    def __init__(self) -> None:
        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('rmse', Float32, queue_size=10)
        self.gt = []
        self.estimated = []
        self.rmse = []
        self.rate = rospy.Rate(1)
        now = datetime.now()
        self.mode = rospy.get_param("/metrics_logger/mode", "")
        self.dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        rospack = rospkg.RosPack()
        self.pkg_path = os.path.abspath(os.path.join(rospack.get_path('rtabmap_ros'), os.pardir))
        plt.figure()

    def run(self):
        while not rospy.is_shutdown():
            try:
                (trans_gt, rot_gt) = self.listener.lookupTransform('/world', '/base_link_gt', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn_throttle(1, "Could not get GT pose. Skipping.")
                self.rate.sleep()
                continue
            try:
                (trans_est, rot_est) = self.listener.lookupTransform('/world', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn_throttle(1, "Could not get estimated pose. Skipping.")
                self.rate.sleep()
                continue
            self.gt.append(trans_gt)
            self.estimated.append(trans_est)
            self.compute_rsme()
            self.plot()
            rospy.loginfo("Saved metrics.")
            self.pub.publish(self.rmse[-1])
            self.rate.sleep()
        rospy.loginfo("No more data received.")

    def compute_rsme(self):
        err = math.sqrt(mean_squared_error(self.gt, self.estimated))
        self.rmse.append(err)

    def plot(self):
        # Plot then save as imageself.mode 
        plt.cla()
        plt.plot(self.rmse, '-r')
        plt.savefig("{}/results/rmse_{}_{}.png".format(self.pkg_path, self.mode, self.dt_string))
        np.save("{}/results/rmse_{}_{}.npy".format(self.pkg_path, self.mode, self.dt_string), self.rmse)


if __name__ == "__main__":
    rospy.init_node('metrics_logger')
    try:
        logger = MetricsLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass