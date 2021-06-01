# -*- coding: utf-8 -*-

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
         'X', 'Y', 'Z', 'O']
numbers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
   'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def Str2PlateNo(plate_str):
    index = np.array(plate_str.split('_'))

    province = provinces[index[0]]
    letter = alphabets[index[1]]
    number = numbers[index[2:]]
    
    return province + letter + number
    
def Str2BoxGraphic(box_data):
    vertex = box_data.split('_')
    upper_left = np.array([int(x) for x in vertex[0].split('&')])
    bottom_right = np.array([int(x) for x in vertex[1].split('&')])
    
    rect = patches.Rectangle(tuple(upper_left), tuple(bottom_right - upper_left)[0], tuple(bottom_right-upper_left)[1], linewidth=1, edgecolor='red', facecolor='none')
    
    return rect

Str2Box = lambda box_data : np.array([int(x) for x in re.split('&|_', box_data)])
NormalizeImg = lambda img : np.array(img / 255.0).astype(np.float32)

def NormalizeBound(arr, x, y):
    
    arr1 = arr[0:3:2] / x
    arr2 = arr[1:4:2] / y
    
    return np.concatenate((arr1, arr2))
    

def PopulateDataset(patharr):
    boxes = []
    plates = []
    imgs = []
    
    count = 1
    for imgp in patharr:
        print(count)
        count=count+1
        imgp_split = imgp.split('\\')[len(imgp.split('\\'))-1].split('-')
        boxes.append(Str2Box(imgp_split[2]))
        plates.append(imgp_split[4].split('_'))
        with PIL.Image.open(imgp) as img:
            imgs.append(np.asarray(img))
    
    return boxes, plates, imgs
        
train_ds_path = "data\\train"
test_ds_path = "data\\test"
val_ds_path = "data\\val"
        
train_img_paths = [train_ds_path + '\\' + p for p in os.listdir(train_ds_path)]
test_img_paths = [test_ds_path + '\\' + p for p in os.listdir(test_ds_path)]
val_img_paths = [val_ds_path + '\\' + p for p in os.listdir(val_ds_path)]

train_boxes, train_plates, train_imgs = PopulateDataset(train_img_paths)
train_imgs = [NormalizeImg(x) for x in train_imgs]
train_boxes = [NormalizeBound(x, 720.0, 1160.0) for x in train_boxes]

