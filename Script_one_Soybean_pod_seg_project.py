#!/usr/bin/env python
# coding: utf-8

# CODE TO TRAIN PRE-PROCESSING CLASSIFIER

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
from torch.utils.data import random_split
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE=32
num_epochs=100
lr=1e-4
class_size=2


tranform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
tranform_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

torch.manual_seed(2021)

train_loader = torchvision.datasets.ImageFolder(root="E:\\Jithin\\machine_learning_data\\Peas_Image\\Classification_Images", transform=tranform_train)
val_size = 50 
train_size = len(train_loader) - val_size
train_loader, val = random_split(train_loader, [train_size, val_size])
test_loader =torchvision.datasets.ImageFolder(root="E:\\Jithin\\machine_learning_data\\Peas_Image\\Classification_Images", transform=tranform_test)
data_loader_train = torch.utils.data.DataLoader(train_loader, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)\
data_loader_test = torch.utils.data.DataLoader(test_loader, 
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

classes = ('Background', 'Soybean')

sample_image=iter(data_loader_train)
samples,labels=sample_image.next()
print(samples.shape) #64 batch size, 1 channel, width 224 , height 224
print(labels)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(data_loader_train)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break

import math
block1 =224
pool1 =math.ceil((block1-3)/2 +1)
print(pool1)


block2=pool1

pool2 =math.ceil((block2-3)/2 +1)
print(pool2)



block3=pool2
pool3 =math.ceil((block3-3)/2 +1)
print(pool3)


block4=pool3
pool4 =math.ceil((block4-3)/2 +1)
print(pool4)


block5=pool4
pool5 =math.ceil((block5-3)/2 +1)
print(pool5)


#After flatten 
flatten= pool5 * pool5 * 512
print(f'After flatten:: {flatten}')


class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = VGG16_NET() 
model = model.to(device=device) 
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= lr) 

for epoch in range(num_epochs): #I decided to train the model for 50 epochs
    loss_var = 0
    
    for idx, (images, labels) in enumerate(data_loader_train):
        images = images.to(device=device)
        labels = labels.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(images)
        loss = criterion(scores,labels)
        loss.backward()
        optimizer.step()
        loss_var += loss.item()
        if idx%64==0:
            print(f'Epoch [{epoch+1}/{num_epochs}] || Step [{idx+1}/{len(train_loader)}] || Loss:{loss_var/len(train_loader)}')
    print(f"Loss at epoch {epoch+1} || {loss_var/len(train_loader)}")

    with torch.no_grad():
        correct = 0
        samples = 0
        for idx, (images, labels) in enumerate(data_loader_test):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum()
            samples += preds.size(0)
        print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")

# SAVE AND LOAD TRAINED MODEL

torch.save(model.state_dict(), "peas_cleaning_model.pt") #SAVES THE TRAINED MODEL
model = VGG16_NET()
model.load_state_dict(torch.load("peas_cleaning_model.pt")) #loads the trained model
model.eval()


correct = 0
samples = 0
for idx, (images, labels) in enumerate(data_loader_test):
    images = images.to(device='cpu')
    labels = labels.to(device='cpu')
    outputs = model(images)
    _, preds = outputs.max(1)
    correct += (preds == labels).sum()
    samples += preds.size(0)
print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")


# THIS SECTION DEALS WITH TESTING ON IMAGES FROM FOLDER

from PIL import Image
from glob import glob
import os
from tqdm import tqdm

# image = Image.open('E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\2022_soybean_RealSense_data\\combined_images\\20221003_20_43_46_360522_RTS.png')

tranform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# input = tranform(image)

# input = input.view(1, 3, 224,224)
# output = model(input)

# prediction = int(torch.max(output.data, 1)[1].numpy())
# print(prediction)

# if prediction == 1:
#     print("soybean")
# elif prediction == 0:
#     print("background")

# THIS SECTION DEALS WITH DETECTION OR CLASSIFICATION ON MULTIPLE IMAGES FROM A FOLDER

images_sep = glob('E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\'+'*.jpg')
print(len(images_sep))
for _ in tqdm(images_sep):
    __ = _
    _ = Image.open(_)
    input = tranform(_)
    input = input.view(1, 3, 224,224)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    
    if prediction == 1:
        _.save('E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\1\\'+os.path.basename(__))
    elif prediction == 0:
        _.save('E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\0\\'+os.path.basename(__))
    elif prediction == 2:
        _.save('E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\2\\'+os.path.basename(__))
    elif prediction == 3:
        _.save('E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\3\\'+os.path.basename(__))    
    

# THIS CODE TAKES RANDOM N IMAGES FROM A SET OF FOLDERS TO SAVES IN A NEW DESTINATION FOLDER (FOR CREATING NEW DATASETS)

import random as r
from glob import glob
import os
from shutil import copy

source_folders = ['E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\2022_soybean_RealSense_data\\combined_images\\soybean']

image_format = ['.jpg','.png']

source_memory = [z for y in source_folders for z in glob(os.path.join(y+'\\'+"*"+image_format[1]))]

dest_folder = "E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\2022_soybean_RealSense_data\\labelling_sample\\YOLO_training_dataset_stage2"
n=1000
i=0
temp_list = []

while i<=n:
    src = r.choice(source_memory)
    if src not in temp_list:
        temp_list.append(src)
        copy(src, os.path.join(dest_folder,str(i)+image_format[1])) # enter image format from the list
        i+=1

# TEST CODE 

# import os

# def type_return(item):
#      if os.path.isfile(f):
#         return "file"
#     else if os.path.isdir(f):
#         return "dir"

# def return_content(content):return [_ for _ in os.listdir(content)]
    
# file_list = []
    
# directory = 'E:\\Jithin\\machine_learning_data\\Peas_Image\\'
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     for _ in f:
#         content_type = type_return(_)
#         while content_type == 'dir':
#             type_return(os.listdir(_))
        
        
# list(path.glob('*/*/*'))


# In[34]:


# import os
# def list_files(filepath, filetype):
#     paths = []
#     for root, dirs, files in os.walk(filepath):
#         for file in files:
#             if file.lower().endswith(filetype.lower()):
#                 paths.append(os.path.join(root, file))
#     return(paths)

# my_files_list = list_files(directory, '.jpg')

# len(my_files_list)


# CODE TO MOVE FILES FROM ONE LOCATION TO THE OTHER 

import shutil
from tqdm import tqdm
dst_folder = "E:\\Jithin\\machine_learning_data\\peas_Pisum_sativum_images_combined_images\\"

count = 0
for files in tqdm(my_files_list):
    shutil.move(files, os.path.join(dst_folder, str(count)+".jpg"))
    count+=1



# Processing Flax datasets

# SPLIT A SINGLE IMAGE TO 4 HALFS VERTICALLY AND HORIZONTALLY 

from split_image import split_image
from glob import glob

flax_images = glob('E:\\Jithin\\20222_summerGround_data\\Proximal_sensing\\flax\\raw_images\\'+'*.jpg')
len(flax_images)

for _ in range(len(flax_images)):
    if _ % 30 == 0:
        split_image(flax_images[_], 2, 2, False, False, 'E:\\Jithin\\20222_summerGround_data\\Proximal_sensing\\flax\\raw_images\\split\\')


# classify Carrie's field data into rows and plots

from PIL import Image
from glob import glob
import os
from tqdm import tqdm

model = VGG16_NET()
model.load_state_dict(torch.load("soy_background_sep_model.pt")) #loads the trained model
model.eval()

images_sep = glob('E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\2022_soybean_RealSense_data\\RR_4_rows\\4_rows\\10072022_soybeans_casselton_D405\\segmented\\'+'*.png')
print(len(images_sep))

tranform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


folder_lock = False
dir_name = ""

count=0
for _ in tqdm(images_sep):
    __ = _
    _ = Image.open(_)
    input = tranform(_)
    input = input.view(1, 3, 224,224)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    
    if prediction == 0: # background 
        if folder_lock == False:
            dir_name = 'E:\\Jithin\\machine_learning_data\\soybean_realsense_segmenation_complete_dataset\\2022_soybean_RealSense_data\\RR_4_rows\\4_rows\\10072022_soybeans_casselton_D405\\directory\\'+str(count)
            os.mkdir(dir_name)
            count+=1
            folder_lock = True
    elif prediction == 1: # soybean
        _.save(dir_name+'\\'+os.path.basename(__))
        folder_lock = False
