#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
from glob import glob
import numpy as np


# In[23]:


RGB_images = glob('G:\\2022_DATA\\2022_Soybean_Realsense_data\\10032022_soybeans_casselton_D405\\'+'*.png')
Depth_images = glob('G:\\2022_DATA\\2022_Soybean_Realsense_data\\10032022_soybeans_casselton_D405\\original\\'+'*.png')
print(len(RGB_images))


# In[44]:


print(RGB_images[35])
print(Depth_images[34])
image_RGB = cv2.imread(RGB_images[35])
image_depth = cv2.imread(Depth_images[34])
# image_RGB = image_RGB[image_depth > 0]
# cv2.imwrite('mouseRGB.png', image_RGB)
# image_RGB_BCK = image_RGB
image_depth_BCK = image_depth


# In[38]:


# image_RGB = cv2.imread('G:\\2022_DATA\\2022_Soybean_Realsense_data\\10032022_soybeans_casselton_D405\\20221003_20_43_39_902165_RTS.png')
# image_depth = cv2.imread('G:\\2022_DATA\\2022_Soybean_Realsense_data\\10032022_soybeans_casselton_D405\\original\\20221003_20_43_39_902165_ORIG.png')


# In[46]:


temp_depth = np.asanyarray(image_depth)
temp_depth[temp_depth > 5000] = 0
image_RGB = np.asanyarray(image_RGB)
# image_RGB[temp_depth==0] = [255,255,255]
image_RGB[np.all(temp_depth == (0, 0, 0), axis=-1)] = (255,255,255)
image_RGB = np.hstack((image_RGB_BCK,image_RGB, image_depth_BCK))
cv2.imwrite('mouseRGB.png', image_RGB)


# In[45]:


image_RGB_BCK = cv2.imread(RGB_images[35])

