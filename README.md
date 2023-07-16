# Soybean pod count using Depth segmentation approach (project 2022)

<p>Enhancing soybean yield (scientific name: <em>Glycine max</em> L. (<em>Merr.</em>)) is crucial for strengthening national food security. To achieve this, accurately predicting soybean yield before crop maturity is essential. However, traditional methods often face challenges in estimating yield due to issues with the background color of the crops.</p>

<p>In order to overcome this challenge, we investigated the use of a depth camera to filter RGB images in real-time, aiming to improve the performance of the pod-counting classification model. Furthermore, we compared different object detection models, such as YOLOV7 and YOLOv7-E6E, to select the most suitable deep learning (DL) model for accurately counting soybean pods.</p>

<p>After identifying the optimal architecture, we conducted a comparative analysis of the DL model's performance by training it with and without background removal from the images. The results showed that using a depth camera to remove the background significantly improved the pod detection performance of YOLOv7, increasing precision by 10.2%, recall by 16.4%, mAP@50 by 13.8%, and mAP@0.5:0.95 score by 17.7% compared to when the background was present.</p>

<p>By employing the depth camera and the YOLOv7 algorithm for pod detection and counting, we achieved a mAP@0.5 of 93.4% and mAP@0.5:0.95 of 83.9%. These findings clearly demonstrate the substantial enhancement in the DL model's performance when the background was segmented and a reasonably larger dataset was utilized for training YOLOv7.</p>

<div align="center">
  <img src="mouseRGB.png" width="600">
</div>

<div style="text-align: center;">
  [<img src="5.png" width="300">]<--(https://github.com/jithin8mathew/soybean_pod_count_Depth_segmenation_project) -->
</div>

- Platform
  <p>The camera was mounted on a platform at a height of 44.4 cm from ground level to ensure that the full length of the soybean plants was captured in each frame. The sensor placement on the platform remained consistent throughout the data collection, maintaining the same camera angle and field of view for all images. The platform was manually moved across the field during data collection, and image capturing was automated using a Python v3.9.11 script.</p>

<div style="text-align: center;">
  [<img src="platform.png" width="300">](https://github.com/jithin8mathew/soybean_pod_count_Depth_segmenation_project)
</div>

<div style="text-align: center;">
  [<img src="Training_comparison.png" width="600">](https://github.com/jithin8mathew/soybean_pod_count_Depth_segmenation_project)
</div>

- application:

Soybean molecular breeding program
Yield estimation

Methods used:
- depth segmentation
- object detection
- pod counting
- Yield correlation with the ground-truth data
