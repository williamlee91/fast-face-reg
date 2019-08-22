# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:21:21 2019

@author: Administrator
"""

import center_loss as cl
from keras.models import Model 
#from face_train_faceboxes import Model_train
#from face_train9_bigdata import Model_train
#from face_train8_add_128 import Model_train
#from face_train9_9fire import Model_train
#from face_train_MobileNet2 import Model_train
from face_train7 import Model_train


from keras import backend as K
import sys
sys.path.append('E:/sign_system/execute_system')

def half_model():
    
    img_rows = 96
    img_cols = 96
    img_channels = 3
    if K.image_data_format() == 'channels_first':
        input_shape = (img_channels, img_rows, img_cols)            
    else:
        input_shape = (img_rows, img_cols, img_channels)  
    print(input_shape)
    path = 'data'
    path_data = 'data_hdf5_0.99'
    num_class = 8631 
    batch_size = 128
    epoch = 22
    model = Model_train(batch_size, epoch, input_shape, num_class, path_data)
    
    file_path = 'E:/sign_system/execute_system/model7/best_model.h5'
    model.load_model(file_path)
    #print(model.layer(133))
    #new_model = Model(inputs=model.layer(0).input, outputs=model.layer(133).output)
    new_model = Model(inputs=model.layer(0).input, outputs=model.model.get_layer('flatten').output)
    path = 'E:/sign_system/execute_system/haar_extract/half_7.h5'
    new_model.save(path)
    print("model saved")
    
if __name__=="__main__":
    half_model()