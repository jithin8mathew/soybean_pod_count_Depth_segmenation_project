import pyrealsense2 as rs
import os
import sys
from pathlib import Path
import numpy as np
import math
import cv2

from datetime import datetime
import time


pipeline = rs.pipeline()
config = rs.config()
config.enable_device('123622270756')
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
######################################################################
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#####################################################################

pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('127122270289')
config_2.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
######################################################################
config_2.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#####################################################################

print("[INFO] Starting streaming...")
pipeline.start(config)
pipeline_2.start(config_2)
print("[INFO] Camera ready.")

cv2.namedWindow('dualRealSense',cv2.WINDOW_NORMAL)
    
prev = time.time()

##ctx = rs.context()
##devices = ctx.query_devices()
##print(devices[0],devices[1])

## Intel RealSense D405 (S/N: 123622270756  FW: 05.12.14.100  on USB3.2)
## Intel RealSense D405 (S/N: 127122270289  FW: 05.12.14.100  on USB3.2)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth = frames.get_depth_frame()
    
    temp_depth = np.asanyarray(depth.get_data())
    temp_depth[temp_depth > 5000] = 0

    #############################################
    ############ camera 2 #######################

    frames_2 = pipeline_2.wait_for_frames()
    color_frame_2 = frames_2.get_color_frame()
    depth_2 = frames_2.get_depth_frame()
    
    temp_depth_2 = np.asanyarray(depth_2.get_data())
    temp_depth_2[temp_depth_2 > 5000] = 0
    #############################################
    
    if not depth: continue

    color_image = np.asanyarray(color_frame.get_data())
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_color_frame = rs.colorizer().colorize(depth)

    color_image[temp_depth==0] = [255,255,255]

    color_image_ = np.asanyarray(depth_color_frame.get_data())

    #############################################
    ################## camera 2 #################

    color_image_2 = np.asanyarray(color_frame_2.get_data())
    color_intrin_2 = color_frame_2.profile.as_video_stream_profile().intrinsics
    depth_color_frame_2 = rs.colorizer().colorize(depth_2)

    color_image_2[temp_depth_2==0] = [255,255,255]

    color_image__2 = np.asanyarray(depth_color_frame_2.get_data())
    #############################################
    
##    cv2.setMouseCallback('mouseRGB',mouseRGB)

    cur = time.time()
    images = np.vstack((color_image_2,color_image))
   
    cv2.imshow('dualRealSense', images)
##    if cur-prev >= 1:
##            prev = cur
##            print("writing...")
##            cv2.imwrite(os.getcwd()+'\\2022_soybean_RealSense_data\\09172022_soybean_casselton_D405\\'+str(time.strftime("%Y%m%d-%H%M%S").replace('-','_'))+"_RTS.png",color_image)
##            print(os.getcwd()+'\\2022_soybean_RealSense_data\\09172022_soybean_casselton_D405\\'+str(datetime.now())+"_RTS.png")
##            cv2.imwrite(os.getcwd()+'\\2022_soybean_RealSense_data\\09172022_soybean_casselton_D405\\original\\'+str(time.strftime("%Y%m%d-%H%M%S").replace('-','_'))+"_ORIG.png",color_image_)
    cv2.waitKey(1)
