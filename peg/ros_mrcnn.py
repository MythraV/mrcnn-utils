#!/usr/bin/env python
# coding: utf-8

# Code and visualizations to test, debug, and evaluate the Mask R-CNN model.

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")

# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
# print(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import peg
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# ## Configurations

# def get_ax(rows=1, cols=1, size=16):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
    
#     Adjust the size attribute to control how big to render images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#     return ax

class Peg_Detect():
    def __init__(self):
        self.image = np.zeros((640,480))
        self.rec_img = False
        self.bridge = CvBridge()

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      (image[:, :, c] *
                                      (1 - alpha) + alpha * color[c]).astype(np.int),
                                      image[:, :, c])
        return image
        
    def get_masked_image(self, image, boxes, masks, class_ids, class_names,
                          scores=None, title="",
                          figsize=(16, 16), show_mask=True,
                           show_bbox=True, colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # Generate random colors
        colors = colors or visualize.random_colors(N)

        masked_image = image.astype(np.uint8).copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(N):
            if scores[i]>0.9:
                #color = [int(255*j) for j in colors[i]]
                color = [255,0,0]
                print('COLOR ', color)
                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                if show_bbox:
                    cv2.rectangle(masked_image, (x1, y1), (x2, y2), (color[0],color[1],color[2]), 1)
                # Label
                if not captions:
                    class_id = class_ids[i]
                    score = scores[i] if scores is not None else None
                    label = class_names[class_id]
                    caption = "{} {:.3f}".format(label, score) if score else label
                else:
                    caption = captions[i]
                cv2.putText(masked_image,caption,(x1,y1+8), font, 0.5,(255,255,255),1)

                # Mask
                mask = masks[:, :, i]
                if show_mask:
                    masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            #padded_mask = np.zeros(
            #    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            #padded_mask[1:-1, 1:-1] = mask
            #contours = find_contours(padded_mask, 0.5)
            #for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
            #    verts = np.fliplr(verts) - 1
            #    p = Polygon(verts, facecolor="none", edgecolor=color)
        return(masked_image.astype(np.uint8))

    def image_cb(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data,'bgr8')
        self.rec_img = True


if __name__ == '__main__': 
    ROOT_DIR = '/home/isat/realsense_ws/src/Mask_RCNN'
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Path to weights 
    PEG_WEIGHTS_PATH = "/home/isat/realsense_ws/src/Mask_RCNN/logs/peg20190815T1255/mask_rcnn_peg_0010.h5"  

    config = peg.CustomConfig()
    print(ROOT_DIR)
    PEG_DIR = os.path.join(ROOT_DIR, "maskRCNN/Mask_RCNN/samples/peg/dataset")

    # Override the training configurations with a few
    # changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    # Load weights
    print("Loading weights ", PEG_WEIGHTS_PATH)
    model.load_weights(PEG_WEIGHTS_PATH, by_name=True)
    cls_names = ['BG','peg']
    vid_path = '/media/crl/DATA/Datasets/Forward/Yumi/Yumi_NewC.avi'

    pd = Peg_Detect()
    # ROS init
    rospy.init_node('mrcnn_detect', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, pd.image_cb)
    pub = rospy.Publisher('mask', Image, queue_size=10)

    k = 'f'
    succ = True
    cnt = 0
    while not k==ord('q') and succ:
        if pd.rec_img:
            pd.rec_img = False  # reset until next image received
            if cnt%10 == 0:
                cnt = 0 # reset
                img = cv2.cvtColor(pd.image, cv2.COLOR_BGR2RGB)
                results = model.detect([img], verbose=1)
                # Display results
                r = results[0]
                mimg = get_masked_image(imag, r['rois'], r['masks'], r['class_ids'], 
                      	                  cls_names, r['scores'],title="Predictions")
                cv2.imshow('Image', mimg)
                k = cv2.waitKey(1)
            #	display_images([image])
                cnt +=1	
    cv2.destroyAllWindows()

