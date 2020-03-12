import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import argparse
import cv2

def pub_video(vpath, prate, tname):
    '''
        Publish video as ros topic
        @vpath : path to video
        @prate : publish rate
        @tname : published topic name
    '''
    pub = rospy.Publisher(tname, Image, queue_size=10)
    cap = cv2.VideoCapture(vpath)
    vlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('N frames = ',vlength)
    fcnt = 1 # Frame count
    bridge = CvBridge()
    r = rospy.Rate(prate)
    enco = 'passthrough'
    while not rospy.is_shutdown() and fcnt < vlength:
        suc, frame = cap.read()
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enco = 'rgb8'
        rosimg = bridge.cv2_to_imgmsg(frame, enco)
        pub.publish(rosimg)
        fcnt+=1
        r.sleep()


def pub_topic(topic_name, prate, pub_topic_name):
    '''
        Publish video as ros topic
        @topic_name : Name of topic to subscribe to
        @prate : publish rate
        @pub_topic_name : published topic name
    '''
    pub = rospy.Publisher(pub_topic_name, Image, queue_size=10)

    class PubTopic(object):
        def __init__(self):
            self.pub_img = Image()

        def img_cb(self, data):
            self.pub_img = data

    pt = PubTopic()
    rospy.Subscriber(topic_name, Image, pt.img_cb)
    r = rospy.Rate(prate)
    while not rospy.is_shutdown():
        pub.publish(pt.pub_img)
        r.sleep()

def pub_image(ipath, prate, tname):
    '''
        Publish video as ros topic
        @vpath : path to image
        @prate : publish rate
        if prate = 0, it'll be published as latched topic
        @tname : published topic name
    '''
    frame = cv2.imread(ipath)
    enco = 'passthrough'
    if len(frame.shape) > 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enco = 'rgb8'
    bridge = CvBridge()
    if prate==0:
        pub = rospy.Publisher(tname, Image, queue_size=10, latch=True)
        pub.publish(bridge.cv2_to_imgmsg(frame,enco))
        print('Gonna spin')
        rospy.spin()
    else:
        pub = rospy.Publisher(tname, Image, queue_size=10)
        r = rospy.Rate(prate)
        while not rospy.is_shutdown():
            pub.publish(bridge.cv2_to_imgmsg(frame,enco))
            r.sleep()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Publish video (readable with cv2) or republish topics as ROS images')
    parser.add_argument('--video', required=False,
                        metavar="/path/to/video/",
                        help='Path to video file')
    parser.add_argument('--topic', required=False,
                        metavar="/topic_name",
                        help="Name of topic to be republished")
    parser.add_argument('--newtopic', required=False,
                        default='/camera/repub',
                        metavar="/republished_topic_name",
                        help='Name of the publihed topic (default=/camera/repub)')
    parser.add_argument('--image', required=False,
                        metavar="path to image",
                        help='Path to image file to be published as ROS image')
    parser.add_argument('--rate', required=False, default=30,
                        metavar="10",
                        help='Rate for published image (default=30hz)')
    args = parser.parse_args()

    rospy.init_node('image_pub', anonymous=True)
    repub = {'video': pub_video, 'topic': pub_topic, 'image': pub_image}
    # Parse arguments
    # Order of precendence Video > topic > image
    if args.video:  # If video file provided
        repub['video'](args.video, int(args.rate), args.newtopic)
    elif args.topic:
        repub['topic'](args.topic, int(args.rate), args.newtopic)
    elif args.image:
        repub['image'](args.image, int(args.rate), args.newtopic)
