# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:21:56 2019

@author:yuwangwang
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

#把导入的图像的参数确定为96 × 96
IMAGE_SIZE = 96

# 导入了sklearn库的交叉验证模块，利用函数train_test_split()来划分训练集和验证集
# 划分出了20%的数据用于验证，80%用于训练模型    
def split_dataset(images, labels): 

    train_images, valid_images, train_labels, valid_labels = train_test_split(images,\
    labels, test_size = 0.01, random_state = random.randint(0, 100)) 
            
    #print(train_images.shape[0], 'train samples')
    #print(valid_images.shape[0], 'valid samples')    
    return  train_images, valid_images, train_labels ,valid_labels
    
def data2h5(dirs_path, train_images, valid_images, train_labels ,valid_labels):
    
    TRAIN_HDF5 = dirs_path + '/' + "train.hdf5"
    VAL_HDF5 =  dirs_path + '/' + "val.hdf5"
    
    
    #采用标签与图片相同的顺序分别打乱训练集与验证集
    state1 = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state1)
    np.random.shuffle(train_labels)
    
    state2 = np.random.get_state()
    np.random.shuffle(valid_images)
    np.random.set_state(state2)
    np.random.shuffle(valid_labels)
    
    datasets = [
        ("train",train_images,train_labels,TRAIN_HDF5),
        ("val",valid_images,valid_labels,VAL_HDF5)]
    
    for (dType,images,labels,outputPath) in datasets:
        # 初始化HDF5写入
        f = h5py.File(outputPath, "w")
        f.create_dataset("x_"+dType, data=images)
        f.create_dataset("y_"+dType, data=labels)
        #f.create_dataset("x_"+dType, data=images, compression="gzip", compression_opts=9)
        #f.create_dataset("y_"+dType, data=labels, compression="gzip", compression_opts=9)
        f.close()

def read_dataset(dirs):
    
    files = os.listdir(dirs)
    print(files)
    for file in files:
        path = dirs+'/' + file
        dataset = h5py.File(path, "r")
        file = file.split('.')
        set_x_orig = dataset["x_"+file[0]].shape[0]
        set_y_orig = dataset["y_"+file[0]].shape[0]

        print(set_x_orig)
        print(set_y_orig)
    
#循环读取每个标签集下的所有图片
def load_dataset(path_name,data_path):
    images = []
    labels = []
    train_images = []
    valid_images = [] 
    train_labels = []
    valid_labels = []
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
                
        images = np.array(images) 
        t_images, v_images, t_labels ,v_labels  = split_dataset(images, labels) 
        for i in range(len(t_images)):
            train_images.append(t_images[i])
            train_labels.append(t_labels[i])   
        for j in range(len(v_images)):
            valid_images.append(v_images[j])
            valid_labels.append(v_labels[j])
        if counter%50== 49:
            print( counter+1 , "is read to the memory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  
        
        images = []
        labels = [] 
        counter = counter + 1 
    
    print("train_images num: ", len(train_images), "           ", "valid_images num: ",len(valid_images))
    print("start to write all data to hdf5 file ...................................................................")    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
                   
    data2h5(data_path, train_images, valid_images, train_labels ,valid_labels)
    print("File HDF5 id done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    read_dataset(data_path)

if __name__ == '__main__':
    path = "data"
    data_path = "data_hdf5_96_0.99"
    dirs = load_dataset(path,data_path)
    
    
    
    