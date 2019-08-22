# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:32:08 2018

@author: yuwangwang
"""
from keras import backend as K
K.clear_session()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import center_loss as cl

import numpy as np
import h5py

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import center_loss as cl

from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout, Activation, Flatten, Layer
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.layers import AveragePooling2D, Input, Dense, add
from keras.layers import GlobalAveragePooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler
from keras import regularizers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))

IMAGE_SIZE = 96
TRAIN_HDF5 = "/home/yww/hdf5dataset/data_hdf5_96_0.99/train.hdf5"
VAL_HDF5 = "/home/yww/hdf5dataset/data_hdf5_96_0.99/val.hdf5"
weight_decay = 1e-7

      
class Model_train:
    def __init__(self, batch_size, epoch, input_shape, classes, data_path):
        self.model = None
        self.BS = batch_size
        self.epoch = epoch
        self.input_shape = input_shape
        self.classes = classes
        self.binarize = True
        self.data_path = data_path
        print("classes:", self.classes)
    
    def conv_block(self, inputs, filters, block_id, kernel=(3, 3), strides=(1, 1)):
        """
        Normal convolution block performs conv+bn+relu6 operations.
        :param inputs: Input Keras tensor in (B, H, W, C_in)
        :param filters: number of filters in the convolution layer
        :param kernel: kernel size
        :param strides: strides for convolution
        :return: Output tensor in (B, H_new, W_new, filters)
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = Conv2D(filters,
                   kernel,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2(weight_decay),
                   strides=strides,
                   name='conv%d'%(block_id))(inputs)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name='conv%d_bn'%(block_id))(x)
        #x = PReLU('zero', name='conv_prelu%d'%(block_id))(x)
        
        x = Activation('relu', name='conv%d_relu'%(block_id))(x)
        return x
    
    def fire_squeeze(self, inputs, out_channels, block_id):
        """
        Normal fire_squeeze block performs conv+bn+relu6 operations.
        :param inputs: Input Keras tensor in (B, H, W, C_in)
        :param out_channels: number of filters in the convolution layer
        :param block_id: as its name tells
        :return: Output tensor in (B, H_new, W_new, 8*out_channels)
        """
        
        fire_squeeze = Conv2D(out_channels, 
                              (1, 1), 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              kernel_regularizer=l2(weight_decay),
                              padding='same', 
                              name='fire%d_squeeze'%(block_id))(inputs)  
        fire_squeeze_bn= BatchNormalization(epsilon=1e-5, 
                                            momentum=0.9, 
                                            name = 'fire%d_squeeze_bn'%(block_id))(fire_squeeze)
        #fire_squeeze_prelu = PReLU('zeros', name='fire%d_squeeze_prelu'%(block_id))(fire_squeeze_bn)
        fire_squeeze_relu = Activation('relu', name='fire%d_squeeze_relu'%(block_id))(fire_squeeze_bn)
        
        fire_expand1 = Conv2D((out_channels * 4),
                              (1, 1),
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              kernel_regularizer=l2(weight_decay),
                              padding='same', 
                              name='fire%d_expand1'%(block_id))(fire_squeeze_relu) 
        fire_expand1_bn = BatchNormalization(epsilon=1e-5, momentum=0.9, name = 'fire%d_expand1_bn'%(block_id))(fire_expand1)
        #fire_expand1_prelu = PReLU('zeros', name='fire%d_expand1_prelu'%(block_id))(fire_expand1_bn)
        fire_expand1_relu = Activation('relu', name='fire%d_expand1_relu'%(block_id))(fire_expand1_bn)
        
        fire_expand2 = Conv2D((out_channels * 4), 
                              (3, 3), 
                              kernel_initializer='glorot_uniform',
                              bias_initializer='zeros',
                              kernel_regularizer=l2(weight_decay),
                              padding='same', 
                              name='fire%d_expand2'%(block_id))(fire_squeeze_relu)        
        fire_expand2_bn = BatchNormalization(epsilon=1e-5, momentum=0.9, 
                                              name = 'fire%d_expand2_bn'%(block_id))(fire_expand2)
        #fire_expand2_prelu = PReLU('zero', name='fire%d_expand2_prelu'%(block_id))(fire_expand2_bn)
        fire_expand2_relu = Activation('relu', name='fire%d_expand2_relu'%(block_id))(fire_expand2_bn)
        
        x = Concatenate(axis= -1)([fire_expand1_relu, fire_expand2_relu])  
        
        return x
              
    def Squeezefacenet(self):
        
        input_img = Input(shape = self.input_shape)
    
        #y = Input(shape=(self.classes, ))
        #stage 0
        x = self.conv_block(input_img, 48, 1)
        x = self.conv_block(x, 96, 2, strides=(2, 2))
        
        #stage 1
        x1 = self.fire_squeeze(x, 16, 1)      
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool1')(x1)
        
        #stage 2
        x2 = self.fire_squeeze(x1, 32, 2)      
 
        #stage 3
        x3 = self.fire_squeeze(x2, 32, 3)  
        merge2_3 = add([x2, x3], name = "add2_3") 

        #stage 4
        x4 = self.fire_squeeze(merge2_3, 32, 4)                
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool4')(x4)
        
        #stage 5
        x5 = self.fire_squeeze(x4, 32, 5)               
        merge4_5 = add([x4, x5], name = "add4_5")  

        #stage 6
        x6 = self.fire_squeeze(merge4_5, 32, 6)            
        
        #stage 7
        x7 = self.fire_squeeze(x6, 32, 7)  
        merge6_7 = add([x6, x7], name = "add6_7") 
        
        #stage 8
        x8 = self.fire_squeeze(merge6_7, 32, 8)                 
        x8 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool8')(x8)
                
        #stage 9
        x9 = self.fire_squeeze(x8, 32, 9)
        merge8_9 = add([x8, x9], name = "add8_9") 
        
        #GDC
        x10 = SeparableConv2D(256, 
                              (6, 6), 
                              strides=(1, 1), 
                              depth_multiplier=1,
                              activation=None,
                              depthwise_initializer='glorot_uniform',
                              pointwise_initializer='glorot_uniform',
                              name = 's_conv')(merge8_9)
        x10 = BatchNormalization(epsilon=1e-5, momentum=0.9, name = 'sconv_bn')(x10)
        #x10 = PReLU('zero', name='sconv_prelu')(x10)
        #x10 = Activation('relu', name='sconv_relu')(x10)
        #x10 = self.conv_block(x10, 128, 3, kernel=(1, 1))
#        x10 = Conv2D(128,
#                    (1, 1),
#                    strides=1,
#                    padding='same',
#                    activation = None,
#                    kernel_initializer='glorot_uniform',
#                    bias_initializer='zeros',
#                    kernel_regularizer=l2(weight_decay),
#                    name='conv_out')(x10)
#        x10 = BatchNormalization(epsilon=1e-5, momentum=0.9, name = 'conv10_bn')(x10)
        #out
        #out_drop = Dropout(0.5, name='dropout')(x10)
        out_flatten = Flatten(name='flatten')(x10)
        
        #out = ArcFace(self.classes, regularizer=l2(weight_decay))([out_flatten, y])
        out = Dense(self.classes, kernel_initializer='glorot_uniform', name='dense')(out_flatten)      
        
        self.model = Model(inputs = input_img, outputs = out)
        
        #plot_model(self.model, to_file='model/squeezenetface7_bigdata.png', show_shapes=True)
       
        #self.model.summary()
               
    def scheduler(self, epoch):
    # warmn-up
        epochs = [1, 2, 3, 4, 10, 14, 17, 20]
        lr_inputs = [0.04, 0.06, 0.08, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        if epoch in epochs:
            index_lr = epochs.index(epoch)
            lr_now = lr_inputs[index_lr]
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr_now)
            print("pre_lr {}".format(lr))
            print("lr changed to {}".format(lr_now))
        return K.get_value(self.model.optimizer.lr)  
        
    # model train
    def train(self, data_path):        
                
        optimizer = Adam(lr=0.1, beta_1= 0.9, beta_2= 0.999, epsilon= 0.1)
        #optimizer = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)        
        #optimizer= RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-6)
 
        #self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer,metrics=['accuracy'])  
        
        self.model.compile(loss = cl.loss, optimizer = optimizer,metrics=[cl.categorical_accuracy])  
        sess = K.get_session()
        sess.run(tf.global_variables_initializer())
        
        self.model.summary()
        
        #checkpoint_epoch = ModelCheckpoint("model/best_weights8_96.h5", 
        #                                   monitor='categorical_accuracy', 
        #                                   save_best_only=True,
        #                                   mode='max', period=1)
        #callbacks_list = [checkpoint]  
        
        #two wats to save models
        checkpoint_epoch = EveryEpochCheckpoint(self.model)
        checkpoint_best = HignestAccCheckpoint(self.model, monitor='val_categorical_accuracy')
             
        #set learning rate schedule 
        reduce_lr = LearningRateScheduler(self.scheduler)
        callbacks_list = [checkpoint_epoch ,checkpoint_best, reduce_lr]  
        
        db_t = h5py.File(TRAIN_HDF5)
        numImages_t = db_t['y_train'].shape[0] 
        db_v = h5py.File(VAL_HDF5)
        numImages_v = db_v['y_val'].shape[0] 
        db_t.close()
        db_v.close()
        
        print("train num:", numImages_t, "val num:",numImages_v)
        trainGen  = self.generator(True, "train")
        valGen = self.generator(False, "val")
        
        history = self.model.fit_generator(trainGen,
                                           steps_per_epoch = numImages_t//self.BS,
                                           epochs = self.epoch,
                                           shuffle = True,
                                           validation_data = valGen,
                                           validation_steps = numImages_v // 64,
                                           max_queue_size = 64 * 2,
                                           verbose = 1,
                                           callbacks = callbacks_list)
        
        
        categorical_accuracy = history.history['categorical_accuracy']
        val_categorical_accuracy = history.history['val_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(categorical_accuracy))
        plt.plot(epochs, categorical_accuracy, 'r-')
        plt.plot(epochs, val_categorical_accuracy, 'b')
        plt.title('Training and validation accuracy') 
        plt.savefig('model8_add_256/acc8_big.png')
        plt.close()

        plt.plot(epochs, loss, 'r-')
        plt.plot(epochs, val_loss, 'b-')
        plt.title('Training and validation loss')
        plt.savefig('model8_add_256/loss8_big.png')
        
        
#        acc = history.history['acc']
#        val_acc = history.history['val_acc']
#        loss = history.history['loss']
#        val_loss = history.history['val_loss']
#        epochs = range(len(acc))
#        plt.plot(epochs, acc, 'r-')
#        plt.plot(epochs, val_acc, 'b')
#        plt.title('Training and validation accuracy')
#        plt.savefig('model8/acc8_bigdata.png')
#        plt.close()
#        
#        plt.plot(epochs, loss, 'r-')
#        plt.plot(epochs, val_loss, 'b-')
#        plt.title('Training and validation loss')
#        plt.savefig('model8/loss8_bigdata.png') 
        
    def get_train_batch(self, X_train, y_train, batch_size):

        while 1:
            for i in range(0, len(X_train), batch_size):
                if (i+batch_size) >= (len(X_train) - 1):
                    x = X_train[i:(len(X_train) - 1)]
                    y = y_train[i:(len(X_train) - 1)]
                else:    
                    x = X_train[i:i+batch_size]
                    y = y_train[i:i+batch_size]

                yield({'input_1': x}, {'dense': y})        
                                
                
    def generator(self, datagen, mode):
        
        passes=np.inf
        aug = ImageDataGenerator(
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
        
        epochs = 0              
        
        while epochs < passes:
            
            db_t = h5py.File(TRAIN_HDF5)
            db_v = h5py.File(VAL_HDF5)
            numImages_t = db_t['y_train'].shape[0] 
            numImages_v = db_v['y_val'].shape[0]
            
            if mode == "train":                
                for i in np.arange(0, numImages_t, self.BS):
                    images = db_t['x_train'][i: i+self.BS]
                    labels = db_t['y_train'][i: i+self.BS]                    
                    if K.image_data_format() == 'channels_first':
                       images = images.reshape(images.shape[0], 3, IMAGE_SIZE,IMAGE_SIZE) 
                    else:
                       images = images.reshape(images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3) 
                    
                    images = images.astype('float32')
                    images = images - 127.5
                    images = images * 0.0078125
                    
                    if datagen :
                        (images, labels) = next(aug.flow(images, labels, batch_size = self.BS))                        

                    #one-hot code
                    if self.binarize:
                       labels = np_utils.to_categorical(labels,self.classes)  
                          
                    yield ({'input_1': images}, {'dense': labels})
                                          
            elif mode == "val":
                for j in np.arange(0, numImages_v, self.BS):
                    images = db_v['x_val'][j: j+self.BS]
                    labels = db_v['y_val'][j: j+self.BS] 
                                        
                    if K.image_data_format() == 'channels_first':
                
                       images = images.reshape(images.shape[0], 3, IMAGE_SIZE,IMAGE_SIZE) 
                    else:
                       images = images.reshape(images.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3) 
                    
                    images = images.astype('float32')
                    images = images - 127.5
                    images = images * 0.0078125                      
                    
                    if datagen :
                        (images, labels) = next(aug.flow(images, labels, batch_size = self.BS))                      
                    #one-hot code
                    if self.binarize:
                       labels = np_utils.to_categorical(labels, self.classes)  
                          
                    yield ({'input_1': images }, {'dense': labels})
                                       
            epochs += 1
            
    # save model                             
    def save_model(self, file_path):
        self.model.save(file_path)
 
    # load model
    def load_model(self, file_path):
        self.model = load_model(file_path, custom_objects={'loss': cl.loss,'accuracy':cl.categorical_accuracy})
    
    # evaluate model  
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))    
     
    def layer(self, num):
        return self.model.layers[num]
   
   # face reg
    def face_predict(self, image):    
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):                          
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            
        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
              image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    
    
        # normlize to (0,1)
        image = image.astype('float32')
        image /= 255.0
        
        # all probabilities of labels 
        result = self.model.predict(image)
        #print('result:', result )
        
        #find biggest label       
        max_index = np.argmax(result) 

        #第一个参数为概率最高的label的index,第二个参数为对应概率
        return max_index,result[0][max_index] 

class ArcFace(Layer):
    '''
    :param x: the input embedding vectors
    :param y: the input labels vectors
    :param n_classes:  the input labels
    :param s: scalar value default is 64
    :param m: the margin value, default is 0.5
    :return: the final cacualted output
    '''
    def __init__(self, n_classes, s=64, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
    
class EveryEpochCheckpoint(Callback):
    """
    define new callbacks, in order to save model in every epoch
    """
    def __init__(self, model):
        self.model_to_save=model
        
    def on_epoch_end(self, epoch, logs=None):  #log must be defined
        self.model_to_save.save('model8_add_256/model_%d.h5' % epoch)    
      
class HignestAccCheckpoint(Callback):
    """
    define new callbacks, in order to save highest_acc model 
    """
    def __init__(self, model, monitor):
        self.model_to_save=model
        self.monitor = monitor
        self.highest = 0.
        
    def on_epoch_end(self, epoch, logs=None): #log must be defined
        
        acc = logs.get(self.monitor)
        if acc >= self.highest: # save best model
            self.highest = acc
            self.model_to_save.save('model8_add_256/best_model.h5')

        print('acc: %s, highest: %s' % (acc, self.highest))          

"""============================================================================
   1、训练模型和评估模型只能使用一个，顺序采用先训练模型，在进行评估模型
   2、训练完后的模型保存
   3、保存路径为model文件下
"""
if __name__ == '__main__':
    
    img_rows = IMAGE_SIZE 
    img_cols = IMAGE_SIZE
    img_channels = 3
    
    if K.image_data_format() == 'channels_first':
        input_shape = (img_channels, img_rows, img_cols)            
    else:
        input_shape = (img_rows, img_cols, img_channels)  
    print(input_shape)
    path = '/home/yww/data'
    path_data = '/home/yww/hdf5dataset/data_hdf5_96_0.99'
    labels = os.listdir(path)
    num_class = len(labels)
    
    batch_size = 128
    epoch = 22
    model = Model_train(batch_size, epoch, input_shape, num_class, path_data)
    
    if not os.path.exists('model8_add_256'):
        os.makedirs('model8_add_256')
        
    print('\nTrain_Starting-------------------------------------------------------------------------------')
    model.Squeezefacenet()  
    model.train(path_data)
    
    print('Model Saved.')
    model.save_model(file_path = 'model8_add_256/model_last.h5')
    
    K.clear_session()
   
#    model = Model_train()
#    print('\nTesting---------------------------------------------------------')
#    file_path = 
#    model.load_model(file_path)
#    model.evaluate(dataset)
   

