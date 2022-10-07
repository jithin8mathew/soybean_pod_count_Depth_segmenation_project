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

import pandas as pd
from datetime import datetime
import time

# In[ ]:
temp_dist = []
combined_temp_list = {}

def calculate_distance(x,y):
        color_intrin = color_intrin
        ix,iy = self.ix, self.iy
        udist = self.depth_frame.get_distance(ix,iy)
        vdist = self.depth_frame.get_distance(x, y)
        #print udist,vdist

        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
        point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
        #print str(point1)+str(point2)

        dist = math.sqrt(
            math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
                point1[2] - point2[2], 2))
        #print 'distance: '+ str(dist)
        return 

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = color_image[y,x,0]
        colorsG = color_image[y,x,1]
        colorsR = color_image[y,x,2]
        print('x',x,'y',y)
        zDepth = depth.get_distance(y,x)
        # temp_dist.append(zDepth)
        # if len(temp_dist) == 2:
        #     calculateDist(temp_dist)
        #     temp_dist=[]
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
    
prev = time.time()

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth = frames.get_depth_frame()
    
    temp_depth = np.asanyarray(depth.get_data())
    temp_depth[temp_depth > 5000] = 0
    # df = pd.DataFrame(temp_depth)
    # df.to_csv('depthTest.csv')
    # break
    # df = pd.DataFrame(depth)
    # df.to_csv('test.csv', sep='\t')
    # break

    if not depth: continue

    color_image = np.asanyarray(color_frame.get_data())
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_color_frame = rs.colorizer().colorize(depth)

##    color_image[temp_depth==0] = [255,255,255]

    color_image_ = np.asanyarray(depth_color_frame.get_data())
    color_image_ = cv2.rotate(color_image_, cv2.ROTATE_90_CLOCKWISE)
    
    #color_image_[temp_depth==0] = [255,255,255]
    color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
    # color_image[color_image > 300] = 0

    # print(np.argmax(color_image))
    # df = pd.DataFrame(color_image[...,0])
    # df.to_csv('test.csv')
    # break
    cv2.setMouseCallback('mouseRGB',mouseRGB)

    cur = time.time()
    
    cv2.imshow('mouseRGB', color_image)
    if cur-prev >= 0.25:
            prev = cur
##            print("writing...")
            cv2.imwrite(os.getcwd()+'\\2022_soybean_RealSense_data\\10032022_soybeans_casselton_D405\\'+str(datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f").replace('-','_').replace('.','_').replace(':','_'))+"_RTS.png",color_image)
##            print(os.getcwd()+'\\2022_soybean_RealSense_data\\10032022_soybeans_casselton_D405\\'+str(datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f").replace('-','_'))+"_RTS.png")
            cv2.imwrite(os.getcwd()+'\\2022_soybean_RealSense_data\\10032022_soybeans_casselton_D405\\original\\'+str(datetime.utcnow().strftime("%Y%m%d-%H:%M:%S.%f").replace('-','_').replace('.','_').replace(':','_'))+"_ORIG.png",color_image_)
    cv2.waitKey(1)

# realtimePipelineRead()
