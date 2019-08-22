# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:46:12 2018

@author: yuwangwang
"""

"""============================================================================
   2018.2.16版本一，单人单目标   
   1、首先按类别分出采集的数据，比如设置两个文件夹分别为"me"与"other"
   2、"me"设为1，"other"设为0
   
   2018.2.21版本二，多人单目标
   1、首先按类别分出采集的数据，比如设置多个文件夹分别为"a"、"b"、"c"、"d"等
   2、按照分类的人数分别指定序号，比如四个人，则为"0"、"1"、"2"、"3"
   3、设置一个list用来保存子文件夹的文字，用来显示在摄像头上

"""

import os
import sys
import numpy as np
import cv2

#把导入的图像的参数确定为224 × 224
IMAGE_SIZE = 112

# 按照指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    # 获取图像尺寸
    h, w, _ = image.shape
    
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)    
    
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    # RGB颜色
    BLACK = [0, 0, 0]
    
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))
    
#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def load_dataset(path_name):
    images = []
    labels = []
    counter = 0

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path_name):
         child_path = os.path.join(path_name, child_dir)

         for dir_image in  os.listdir(child_path):
             if dir_image.endswith('.jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(resized_img)
                labels.append(counter)

         counter += 1

    # 返回的img_list转成了 np.array的格式
    images = np.array(images)

    return images, labels, counter

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path_name):
    name_list = []
    for child_dir in os.listdir(path_name):
        name_list.append(child_dir)
    return name_list

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, labels, counter= load_dataset(sys.argv[1])
        print(counter)
