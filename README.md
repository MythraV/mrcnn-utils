# Vision setup Procedure

## Mask-RCNN Workspace setup
Setup a virtual environment for maskrcnn with python 3.x and install required libraries
```bash
conda create -n maskenv python=3.6
conda activate maskenv
pip install opencv-contrib-python
```

### Python3 support for ROS
This need some compiling from source and some setup changes

Create a mask_ws

Only issue with setting up python3 support for ROS distros before noetic is cv_bridge  
To do this, follow [this link](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674).

Issues: If you experience issues during catkin build regarding boost-python3  
Then a dirty fix is to use the boost for active python.

Edit the following line in mask_ws/src/vision_opencv/cv_bridge/CMakeLists.txt  
Line 11  find_package(Boost REQUIRED python3) ->  find_package(Boost REQUIRED python)

## Mask-RCNN messages setup
Download the mrcnn_msgs from this [link](https://github.com/MythraV/mrcnn_msgs.git)
```bash
cd ~/mask_ws/src
git clone https://github.com/MythraV/mrcnn_msgs.git
cd ..
catkin build mrcnn_msgs
```

## Mask-RCNN setup
Create a package for maskrcnn and download the main Mask_RCNN files into the package  
Ensure the virtualenv is active before doing this
```bash
cd ~/mask_ws/src
catkin_create_pkg mrcnn_utils rospy std_msgs
git clone https://github.com/matterport/Mask_RCNN.git
cd Mask_RCNN
pip install -r requirements.txt
python3 setup.py install
cd ..
git clone https://github.com/MythraV/mrcnn-utils.git
```
The requirements.txt has tensorflow as one of the requirements.  
Ensure tensorflow version you are installing is compatible with system CUDA version
and version < 2.0

Edit the following lines pub_obj_masks.py in mask_utils/mask-utils:  

Line 11 PKG_DIR='abspath/to/package/dir' Ex: '/home/isat/Forward/mask_ws/src/mrcnn_utils'

Line 12 WEIGHTS_FNAME='name of weights file' Ex: 'mask_rcnn_peg_0009.h5'

Edit the file mrcnn_utils/CMakeLists.txt  
```bash
# At Line 161 uncomment and modify as follows:
 install(PROGRAMS  
   mrcnn-utils/pub_obj_masks.py  
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}  
 )  
```
## Build and run

```bash
cd ~/mask_ws
catkin build mrcnn_utils
# To run
rosrun mrcnn_utils pub_obj_masks.py
```








