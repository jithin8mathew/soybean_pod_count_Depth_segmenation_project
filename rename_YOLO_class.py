#!/usr/bin/env python
# coding: utf-8

# Python code to read a bunch of .txt annotations (YOLO) and replace a specific class no with another class number 

from glob import glob
from tqdm import tqdm

texts = glob('path_to_annotations\\'+'*.txt')
print(len(texts))

for _ in tqdm(texts):
    annots = """"""
    with open(_) as t:
        annotations = t.readlines()
        for cords in annotations:
            annots+= '0 '+cords.split(' ')[1]+' '+cords.split(' ')[2]+' '+cords.split(' ')[3]+' '+cords.split(' ')[4]
    with open(_,'w+') as w:
        w.write(annots)