# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:32:08 2018

@author: yuwangwang
"""

# random() 方法返回随机生成的一个实数，它在[0,1)范围内
import random
import numpy as np
import matplotlib.pyplot as plt

# 为了保存最后的h5文件
import h5py
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.layers import AveragePooling2D, Input, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

# 对load_face_dataset的数据集进行导入
from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE

#指定GPU0，已知totalMemory: 1.96GiB freeMemory: 1.57GiB
#所以最大选取0.7，一般0.5或者0.6即可
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
set_session(tf.Session(config=config))


"""============================================================================
   1、按照交叉验证的原则将数据集划分成三部分：训练集、验证集、测试集；
   2、按照keras库运行的后端系统要求改变图像数据的维度顺序；
   3、将数据标签进行one-hot编码，使其向量化
   4、归一化图像数据         
"""

class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        
        # 测试集
        self.test_images  = None            
        self.test_labels  = None
        
        # 数据集加载路径
        self.path_name    = path_name
        
        # 当前库采用的维度顺序
        self.input_shape = None
        
    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, 
             img_channels = 3):
        # 加载数据集到内存
        images, labels , nb_classes= load_dataset(self.path_name)        
        
        # 导入了sklearn库的交叉验证模块，利用函数train_test_split()来划分训练集和验证集
        # 划分出了30%的数据用于验证，70%用于训练模型
        train_images, valid_images, train_labels, valid_labels = train_test_split(images,\
        labels, test_size = 0.3, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5,\
        random_state = random.randint(0, 100))                
        
        # 当前的维度顺序如果为'channels_first'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)            
            
            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            # 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
            # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            # 将其归一化,图像的各像素值归一化到0~1区间，数据集先浮点后归一化的目的是提升网络收敛速度，
            # 减少训练时间，同时适应值域在（0,1）之间的激活函数，增大区分度
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels

"""============================================================================            
    1、采用Model结构
    2、训练的先行条件时较大的数据集，数据集不大易导致准确率较低
    3、调参时注意学习率以及batch的选择
"""           
class Model_train:
    def __init__(self):
        self.model = None 
       
    # 建立模型        
    def Squeezenet(self, nb_classes, dataset):
                
        input_img = Input(shape = dataset.input_shape)

        conv1 = Conv2D(
            96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', strides=(2, 2), padding='same',  
            name='conv1')(input_img)
        
        maxpool1 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name = 'maxpool1')(conv1)
        
        fire2_squeeze = Conv2D(
            16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire2_squeeze')(maxpool1)
        
        fire2_expand1 = Conv2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire2_expand1')(fire2_squeeze)
        
        fire2_expand2 = Conv2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire2_expand2')(fire2_squeeze)
        
        #注意坐标轴选-1
        merge2 = Concatenate(axis= -1)([fire2_expand1, fire2_expand2])

        fire3_squeeze = Conv2D(
            16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire3_squeeze')(merge2)
        
        fire3_expand1 = Conv2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire3_expand1')(fire3_squeeze)
        
        fire3_expand2 = Conv2D(
            64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire3_expand2')(fire3_squeeze)
        
        merge3 = Concatenate(axis= -1)([fire3_expand1, fire3_expand2])

        fire4_squeeze = Conv2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire4_squeeze')(merge3)
        
        fire4_expand1 = Conv2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire4_expand1')(fire4_squeeze)
        
        fire4_expand2 = Conv2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',            
            padding='same', name='fire4_expand2')(fire4_squeeze)
        
        merge4 = Concatenate(axis= -1)([fire4_expand1, fire4_expand2])
        
        maxpool4 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

        fire5_squeeze = Conv2D(
            32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire5_squeeze')(maxpool4)
        
        fire5_expand1 =Conv2D(
            128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire5_expand1')(fire5_squeeze)
        
        fire5_expand2 = Conv2D(
            128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire5_expand2')(fire5_squeeze)
        
        merge5 = Concatenate(axis= -1)([fire5_expand1, fire5_expand2])

        fire6_squeeze = Conv2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire6_squeeze')(merge5)
        
        fire6_expand1 = Conv2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire6_expand1')(fire6_squeeze)
        
        fire6_expand2 = Conv2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire6_expand2')(fire6_squeeze)
        
        merge6 = Concatenate(axis= -1)([fire6_expand1, fire6_expand2])
                
        fire7_squeeze = Conv2D(
            48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire7_squeeze')(merge6 )
            
        fire7_expand1 = Conv2D(
            192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire7_expand1')(fire7_squeeze)
        
        fire7_expand2 = Conv2D(
            192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire7_expand2')(fire7_squeeze)
        
        merge7 = Concatenate(axis= -1)([fire7_expand1, fire7_expand2])

        fire8_squeeze = Conv2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire8_squeeze')(merge7)
        
        fire8_expand1 = Conv2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire8_expand1')(fire8_squeeze)
        
        fire8_expand2 = Conv2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire8_expand2')(fire8_squeeze)
        
        merge8 = Concatenate(axis= -1)([fire8_expand1, fire8_expand2])

        maxpool8 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)
            
        fire9_squeeze = Conv2D(
            64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire9_squeeze')(maxpool8)
        
        fire9_expand1 = Conv2D(
            256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            padding='same', name='fire9_expand1')(fire9_squeeze)
        
        fire9_expand2 = Conv2D(
            256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='same', name='fire9_expand2')(fire9_squeeze)
        
        merge9 = Concatenate(axis= -1)([fire9_expand1, fire9_expand2])

        fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
        
        conv10 = Conv2D(
            nb_classes, (1, 1), kernel_initializer='glorot_uniform',
            bias_initializer='zeros', 
            padding='valid', name='conv10')(fire9_dropout)

        avgpool10 = AveragePooling2D((6, 6), name='avgpool10')(conv10)
        
        flatten = Flatten(name='flatten')(avgpool10)
        
        softmax = Activation("softmax", name='softmax')(flatten)
    
        self.model = Model(inputs = input_img, outputs = softmax)
        
        self.model.summary()
                                
        #return self.model
        
        
    # 训练模型
    def train(self, dataset, batch_size = 24, epochs = 30, data_augmentation = True):        
        
        # 采用SGD + momentum的优化器进行训练，首先生成一个优化器对象
        # momentum指定动量值，让优化器在一定程度上保留之前的优化方向，同时利用当前样本微调最终的
        # 优化方向，这样即能增加稳定性，提高学习速度，又在一定程度上避免了陷入局部最优陷阱
        # 参数其值为0~1之间的浮点数。一般来说，选择一个在0.5 ~ 0.9之间的数即可
        # 代码中SGD函数的最后一个参数nesterov则用于指定是否采用nesterov动量方法
        # nesterov momentum是对传统动量法的一个改进方法，其效率更高
        # 并完成实际的模型配置工作
        
        optimizer = Adam(lr=0.0001, beta_1= 0.9, beta_2= 0.999, epsilon= 1e-8)
        #optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)        
        
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer,
                           metrics=['accuracy'])   
        
        # 不使用数据提升，所谓的提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:            
            history = self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           epochs = epochs,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        # 使用实时数据提升
        else:            
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center = False,             # 是否使输入数据去中心化（均值为0），
                samplewise_center  = False,             # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization = False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization  = False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening = False,                  # 是否对输入数据施以ZCA白化
                rotation_range = 20,                    # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range  = 0.2,               # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range = 0.2,               # 同上，只不过这里是垂直
                horizontal_flip = True,                 # 是否进行随机水平翻转
                vertical_flip = False)                  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理
            datagen.fit(dataset.train_images)                        

            # 利用生成器开始训练模型
            history = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size = batch_size),
                                                  steps_per_epoch = dataset.train_images.shape[0],
                                                  epochs = epochs,
                                                  validation_data = (dataset.valid_images, dataset.valid_labels))
            
            
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r-')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'r-')
        plt.plot(epochs, val_loss, 'b-')
        plt.title('Training and validation loss')
        plt.show()

    # 一个函数用于保存模型，一个函数用于加载模型。
    # keras库利用了压缩效率更高的HDF5保存模型，所以我们用“.h5”作为文件后缀                                 
    #MODEL_PATH = 'E:/sign_system/execute_system/FaceBoxes/faceregmodel/squeezenetfaceboxes.h5'
    def save_model(self, file_path):
        self.model.save(file_path)
 
    # 导入训练好的模型
    def load_model(self, file_path):
        self.model = load_model(file_path)
    
    # 模型评估   
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))    
     
    def layer(self, num):
        return self.model.layers[num]
   
   # 识别人脸
    def face_predict(self, image):    
        # 依然是根据后端系统确定维度顺序
        # 尺寸必须与训练集一致都应该是IMAGE_SIZE x IMAGE_SIZE
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                            
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
              image = resize_image(image)
              image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    

        #image = img.reshape((1, 1, IMAGE_SIZE, IMAGE_SIZE))     
        # 浮点并归一化
        image = image.astype('float32')
        image /= 255.0
        
        # 则该函数会给出输入图像属于各个目标的概率
        result = self.model.predict(image)
        #print('result:', result )
        
        #找出概率最高的        
        max_index = np.argmax(result) 

        #第一个参数为概率最高的label的index,第二个参数为对应概率
        return max_index,result[0][max_index] 
           

"""============================================================================
   1、训练模型和评估模型只能使用一个，顺序采用先训练模型，在进行评估模型
   2、训练完后的模型保存
   3、保存路径为model文件下
"""


if __name__ == '__main__':
    
    # 读取数据
    dataset = Dataset(r'E:\sign_system\face_asia_500_crop') 
    # 读取路径    
    path_name = r'E:\sign_system\face_asia_500_crop'
    dataset.load()
    model = Model_train()
    
    # 训练模型  
    _,  _, num_classes = load_dataset(path_name) 
    print(num_classes)
    
    print('\nTrain_Starting--------------------------------------------------')
    model.Squeezenet(num_classes, dataset)
      
    model.train(dataset)
    
    print('Model Saved.')
    model.save_model(file_path = 'E:/sign_system/execute_system/haar_extract/squeezenetface4.h5')
    
   
    # 评估模型
#    model = Model_train()
#    print('\nTesting---------------------------------------------------------')
#    model.load_model(file_path = 'C:\Users\Administrator\Desktop\FaceRecognition_Version3\model\squeezenet.model.h5')
#    model.evaluate(dataset)
   

