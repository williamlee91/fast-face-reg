# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:46:12 2018

@author: yuwangwang
"""


import os
import numpy as np
import cv2
import random
from scipy import misc
import h5py
from sklearn.model_selection import train_test_split
from keras import backend as K
K.clear_session()
from keras.utils import np_utils

#把导入的图像的参数确定为224 × 224
IMAGE_SIZE = 128

    
# 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
def split_dataset(images, labels): 
    # 导入了sklearn库的交叉验证模块，利用函数train_test_split()来划分训练集和验证集
    # 划分出了10%的数据用于验证，90%用于训练模型
    train_images, valid_images, train_labels, valid_labels = train_test_split(images,\
    labels, test_size = 0.2, random_state = random.randint(0, 100)) 
    
#    _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5,\
#    random_state = random.randint(0, 100))  
            
    # 输出训练集、验证集、测试集的数量
    #print(train_images.shape[0], 'train samples')
    #print(valid_images.shape[0], 'valid samples')
#    print(test_images.shape[0], 'test samples')

    # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
    # 类别标签进行one-hot编码使其向量化
#    train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
#    valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
#    test_labels = np_utils.to_categorical(test_labels, nb_classes)                        

    # 像素数据浮点化以便归一化
#    train_images = train_images.astype('float32')            
#    valid_images = valid_images.astype('float32')
#    test_images = test_images.astype('float32')
    
    # 将其归一化,图像的各像素值归一化到0~1区间，数据集先浮点后归一化的目的是提升网络收敛速度，
    # 减少训练时间，同时适应值域在（0,1）之间的激活函数，增大区分度
#    train_images /= 255
#    valid_images /= 255
#    test_images /= 255  
            
    return  train_images, valid_images, train_labels ,valid_labels
    
#    return  train_images, valid_images, test_images, train_labels ,valid_labels, test_labels   

def data2h5(dirs_path, train_images, valid_images, train_labels ,valid_labels):

        
#def data2h5(dirs_path, train_images, valid_images, test_images, train_labels ,valid_labels, test_labels):
    
    TRAIN_HDF5 = dirs_path + '/' + "train.hdf5"
    VAL_HDF5 =  dirs_path + '/' + "val.hdf5"
    #TEST_HDF5 =  dirs_path + '/' + "test.hdf5"
#    datasets = [
#        ("train",train_images,train_labels,TRAIN_HDF5),
#        ("val",valid_images,valid_labels,VAL_HDF5),
#        ("test",test_images,test_labels,TEST_HDF5)]
    
    
    datasets = [
        ("train",train_images,train_labels,TRAIN_HDF5),
        ("val",valid_images,valid_labels,VAL_HDF5)]
    
    for (dType,images,labels,outputPath) in datasets:
        # 初始化HDF5写入
        f = h5py.File(outputPath, "w")
        #f.create_dataset("x_"+dType, data=images, compression="gzip", compression_opts=9)
        #f.create_dataset("y_"+dType, data=labels, compression="gzip", compression_opts=9)
        f.create_dataset("x_"+dType, data=images)
        f.create_dataset("y_"+dType, data=labels)
        f.close()

def read_dataset(dirs):
    files = os.listdir(dirs)
    print(files)
    for file in files:
        path = dirs+'/' + file 
        file_read = os.listdir(path)
        for i in file_read:
            path_read = os.path.join(path, i)
            dataset = h5py.File(path_read, "r")
            i = i.split('.')
            set_x_orig = dataset["x_"+i[0]].shape[0]
            set_y_orig = dataset["y_"+i[0]].shape[0]
            print(set_x_orig)
            print(set_y_orig)
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
    
#循环读取每个标签集下的所有图片
def load_dataset(path_name,data_path):
    images = []
    labels = []
    t_images = []
    v_images = []
    t_labels = []
    v_labels = []
    counter = 0
    allpath = os.listdir(path_name)
    nb_classes = len(allpath)
    print("label_num:    ",nb_classes)
    
    for child_dir in allpath:
        child_path = os.path.join(path_name, child_dir)
        for dir_image in os.listdir(child_path):
            if dir_image.endswith('.jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))               
                image = misc.imresize(img, (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear')
                #resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                images.append(image)
                labels.append(counter)
        
        if ((counter % 4316 == 4315) or (counter == nb_classes - 1)): 
            
            images = np.array(images) 
            t_images, v_images, t_labels ,v_labels  = split_dataset(images, labels)
            
            print("start write images and labels  data...................................................................")           
            num = counter // 5000
            dirs = data_path + "/" + "h5_" + str(num - 1)
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            data2h5(dirs, t_images, v_images, t_labels ,v_labels)
            #read_dataset(dirs)
            print("File HDF5_%d "%num, " id done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
            images = []
            labels = []
            t_images = []
            v_images = []
            t_labels = []
            v_labels = []
        if counter%50 == 49:
            print( counter+1 , "is read in memory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
     
        counter = counter + 1    
    read_dataset(data_path)      

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path_name):
    name_list = []
    for child_dir in os.listdir(path_name):
        name_list.append(child_dir)
    return name_list

if __name__ == '__main__':
    path = "data"
    data_path = "data_hdf5_half"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    load_dataset(path,data_path)
    