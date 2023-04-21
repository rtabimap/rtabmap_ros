import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tf
import rospy
from std_msgs.msg import Float32

class MetricsLogger:

    def __init__(self) -> None:
        self.listener = tf.TransformListener()
        self.pub = rospy.Publisher('rmse', Float32, queue_size=10)
        self.gt = []
        self.estimated = []
        self.rmse = []
        self.rate = rospy.Rate(5)

    def run(self):
        while not rospy.is_shutdown():
            try:
                (trans_gt, rot_gt) = self.listener.lookupTransform('/world', '/base_link_gt', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Could not get GT pose. Skipping.")
                continue
            try:
                (trans_est, rot_est) = self.listener.lookupTransform('/world', '/base_link', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Could not get estimated pose. Skipping.")
                continue
            self.gt.append(trans_gt)
            self.estimated.append(trans_est)
            self.compute_rsme()
            self.pub.publish(self.rmse[-1])
            self.rate.sleep()

    def compute_rsme(self):
        err = math.sqrt(mean_squared_error(self.gt, self.estimated))
        self.rmse.append(err)

    def plot(self):
        # Plot then save as image
        pass


if __name__ == "__main__":
    rospy.init_node('metrics_logger')
    try:
        logger = MetricsLogger()
        logger.run()
    except rospy.ROSInterruptException:
        pass