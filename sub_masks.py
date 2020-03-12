#!/usr/bin/env python
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from mrcnn_msgs.msg import ObjMasks, BoundingBox
import cv2
import numpy as np

def mask_cb(obm):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    if len(obm.masks)>0:
        # for i, m in enumerate(obm.masks):
        img = (CvBridge().imgmsg_to_cv2(obm.masks[0],'passthrough'))/255
        # for msk in obm.masks:
        #    img += CvBridge().imgmsg_to_cv2(msk,'passthrough')/255
        cv2.imshow('image',img)#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('sub_mask',anonymous=True)
    rospy.Subscriber('/masks_t', ObjMasks, mask_cb)
    while not rospy.is_shutdown():
        continue
    cv2.destroyAllWindows()
