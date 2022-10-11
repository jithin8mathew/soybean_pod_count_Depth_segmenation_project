#!/usr/bin/env python
# coding: utf-8

import cv2
from glob import glob
import numpy as np
import os
from tqdm import tqdm

input_folder_root = os.getcwd()+"\\2022_soybean_RealSense_data\\10102022_soybeans_casselton_D405\\"

RGB_images = glob(input_folder_root+'*.png') # store data from RGB folder to a list with file name and file path
Depth_images = glob(input_folder_root+'original\\'+'*.png') # store data from depth image folder to a list with file name and file path
print(len(RGB_images))
print(len(Depth_images))
print(os.path.join(input_folder_root,'\\original\\'))

if len(RGB_images) & len(Depth_images) != 0: # if both RGB files and depth data files are not empty proceed 
    for _ in tqdm(range(len(RGB_images))):             # iterate through images
        image_RGB = cv2.imread(RGB_images[_])
        image_depth = cv2.imread(Depth_images[_])    
        temp_depth = np.asanyarray(image_depth) # convert the image to numpy array
        temp_depth[temp_depth > 5000] = 0 # filter out the depth image values above 5000
        image_RGB = np.asanyarray(image_RGB) 
        # image_RGB[temp_depth==0] = [255,255,255]
        image_RGB[np.all(temp_depth == (0, 0, 0), axis=-1)] = (255,255,255) # filter the color image by filterd out depth value 
        cv2.imwrite(input_folder_root+'segmented\\'+os.path.basename(RGB_images[_]), image_RGB) # write the segmented image with the original image name 

