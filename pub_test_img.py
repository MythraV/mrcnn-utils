#!/usr/bin/env python
import os
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

img_path = '/media/crl/DATA/Mythra/Research/maskRCNN/Mask_RCNN/samples/box/dataset/val/val_0.jpg'
cv_img = cv2.imread(img_path)
ros_img = CvBridge().cv2_to_imgmsg(cv_img)

rospy.init_node('test_img_pub', anonymous=True)
pub = rospy.Publisher('/camera/image/test_rgb', Image, queue_size=10)

r = rospy.Rate(30)
while not rospy.is_shutdown():
    pub.publish(ros_img)
    r.sleep()
