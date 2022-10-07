#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyrealsense2 as rs
import os
import sys
from pathlib import Path
import numpy as np
import math
import cv2

# In[ ]:
temp_dist = []
combined_temp_list = {}

def mouseRGB(event,x,y,flags,param):
    count = 0
    while count % 2 == 0:
        if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
            colorsB = color_image[y,x,0]
            colorsG = color_image[y,x,1]
            colorsR = color_image[y,x,2]
            print('x',x,'y',y)
            zDepth = depth.get_distance(y,x)
            print('Depth',zDepth) 
            colors = color_image[y,x]
            print("Red: ",colorsR)
            print("Green: ",colorsG)
            print("Blue: ",colorsB)
            print("BRG Format: ",colors)
            print("Coordinates of pixel: X: ",x,"Y: ",y)


# def realtimePipelineRead():
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
######################################################################
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
#####################################################################
print("[INFO] Starting streaming...")
pipeline.start(config)
print("[INFO] Camera ready.")

cv2.namedWindow('mouseRGB')
    
    
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth = frames.get_depth_frame()

    if not depth: continue

    color_image = np.asanyarray(color_frame.get_data())
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_color_frame = rs.colorizer().colorize(depth)

    cv2.setMouseCallback('mouseRGB',mouseRGB)
        
    cv2.imshow('mouseRGB', color_image)
    cv2.waitKey(1)

# realtimePipelineRead()

