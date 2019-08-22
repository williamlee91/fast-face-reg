# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:32:08 2018

@author: yuwangwang
"""
from keras import backend as K
K.clear_session()

# random() 方法返回随机生成的一个实数，它在[0,1)范围内
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import math
import numpy as np
import h5py

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#import center_loss as cl
from keras import initializers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout, Activation, Flatten, Layer
from keras.layers import Conv2D, MaxPooling2D, Concatenate
from keras.layers import AveragePooling2D, Input, Dense
from keras.layers import GlobalAveragePooling2D, SeparableConv2D
from keras.layers import Lambda, add
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
weight_decay = 1e-5

      
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
    
    def InvertedResidualBlock(self, x, expand, out_channels, stride, weight_decay, block_id):
        """
        This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.
        :param x: Input Keras tensor in (B, H, W, C_in)
        :param expand: expansion factor in bottlenect residual block
        :param out_channels: number of channels in the output tensor
        :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.
        :param stride: stride for the 1x1 convolution
        :param weight_decay: hyperparameter for the l2 penalty
        :param block_id: as its name tells
        :return: Output tensor (B, H_new, W_new, out_channels)
    
        """
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        in_channels = K.int_shape(x)[channel_axis]
        x = Conv2D(expand * in_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='dwconv_%d_0' % block_id)(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='dwconv_%d_0_bn' % block_id)(x)
        x = Activation('relu', name='dwconv_%d_0_act_1' % block_id)(x)
        x = SeparableConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=stride,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name='dwconv_dw_%d_0' % block_id )(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='dwconv_dw_%d_0_bn' % block_id)(x)
        x = Activation('relu', name='dwconv_%d_0_act_2' % block_id)(x)
        x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                   kernel_regularizer=l2(weight_decay), name='dwconv_bottleneck_%d_0' % block_id)(x)
        x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='dwconv_bottlenet_%d_0_bn' % block_id)(x)

        return x
              
    def Squeezefacenet(self):
        
        input_img = Input(shape = self.input_shape)
        y = Input(shape=(self.classes, ))
        
        #stage 0
        x0 = self.conv_block(input_img, 48, 1)
        x0 = self.conv_block(x0, 96, 2, strides=(2, 2))
        
        #stage 1
        x1 = self.fire_squeeze(x0, 16, 1)      
        x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool1')(x1)
        
        #stage 2
        x2 = self.fire_squeeze(x1, 32, 2)      
        
        
        #stage 3
        x3 = self.fire_squeeze(x2, 32, 3)  
        add2_3 = add([x2, x3], name = "add2_3") 
        
        #stage 4
        x4 = self.fire_squeeze(add2_3, 32, 4)                
        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool4')(x4)
        
        #stage 5
        x5 = self.fire_squeeze(x4, 32, 5)               
        add4_5 = add([x4, x5], name = "add4_5")  

        #stage 6
        x6 = self.fire_squeeze(add4_5, 32, 6)            
        
        #stage 7
        x7 = self.fire_squeeze(x6, 32, 7)  
        add6_7 = add([x6, x7], name = "add6_7") 
        
        #stage 8
        x8 = self.fire_squeeze(add6_7, 32, 8)            
                
        #stage 9
        x9 = self.fire_squeeze(x8, 32, 9)  
        add8_9 = add([x8, x9], name = "add8_9") 
        
        #stage 10
        x10 = self.fire_squeeze(add8_9, 32, 10)                 
        x10 =  MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name = 'maxpool10')(x10)
                
        #stage 11
        x11 = self.fire_squeeze(x10, 32, 11)
        add10_11= add([x10, x11],  name = "add10_11")
        
        #GDC
        x11 = SeparableConv2D(256, 
                              (6, 6), 
                              strides=(1, 1), 
                              depth_multiplier=1,
                              activation=None,
                              depthwise_initializer='glorot_uniform',
                              pointwise_initializer='glorot_uniform',
                              name = 's_conv')(add10_11)
        
        x11 = BatchNormalization(epsilon=1e-5, momentum=0.9, name = 'sconv_bn')(x11)
        
        #out
        out_flatten = Flatten(name='flatten')(x11)
        
        
        out = ArcFaceLoss(self.classes)([out_flatten, y])
        #out = Dense(self.classes, kernel_initializer='glorot_uniform', name='dense')(out_flatten) 
        
        self.model = Model(inputs = [input_img, y], outputs = out)
        
        #plot
        #plot_model(self.model, to_file='model/squeezenetface7_bigdata.png', show_shapes=True)
            
    def scheduler(self, epoch):
    # 每隔一定的epoch，学习率减小为原来的1/10
        #epochs = [3, 6, 9, 12]    
        epochs = [1, 2, 3, 4, 16, 20, 22, 24]
        lr_inputs = [0.04, 0.06, 0.08, 0.01, 0.001, 0.0001, 0.00001]
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
        
        optimizer = Adam(lr=0.1, beta_1= 0.9, beta_2= 0.999, epsilon= 0.1, decay=1e-5)
        #optimizer = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)        
        #optimizer= RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-6)
 
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizer,metrics=['accuracy'])
         
        self.model.summary()
        
        #two wats to save models
        checkpoint_epoch = EveryEpochCheckpoint(self.model)
        checkpoint_best = HignestAccCheckpoint(self.model, monitor='val_acc')
             
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
                
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r-')
        plt.plot(epochs, val_acc, 'b')
        plt.title('Training and validation accuracy')
        plt.savefig('model9_11fire/acc9_bigdata.png')
        plt.close()
        
        plt.plot(epochs, loss, 'r-')
        plt.plot(epochs, val_loss, 'b-')
        plt.title('Training and validation loss')
        plt.savefig('model9_11fire/loss9_bigdata.png') 
        
        
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
                          
                    yield ({'input_1': images, 'input_2': labels}, {'arc_face_loss_1': labels})
                                          
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
                          
                    yield ({'input_1': images, 'input_2': labels}, {'arc_face_loss_1': labels})
                                       
            epochs += 1
            
    # save model                             
    def save_model(self, file_path):
        self.model.save(file_path)
 
    # load model
    def load_model(self, file_path):
        self.model = load_model(file_path, custom_objects={'ArcFaceLoss': ArcFaceLoss})
    
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
    
    
class ArcFaceLoss(Layer):
    '''
    :param class_num:  the input labels 
    :param s: scalar value default is 64
    :param m: the margin value, default is 0.5
    :return: the final cacualted output
    '''
    def __init__(self, class_num = 8631, s=64, m=0.5, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.class_num = class_num
        self.s = s
        self.m = m
        super(ArcFaceLoss, self).__init__(**kwargs)

    '''   
    used to create the weight vector 
    which will be learned of this layer
    op add_weight: add trainble parameter
    param input_shape : first input_shape
    '''
    def build(self, input_shape):
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight((input_shape[0][-1], self.class_num), initializer=self.init,
                                 name='{}_W'.format(self.name), regularizer=l2(weight_decay))
        super(ArcFaceLoss, self).build(input_shape)

    '''
    defibe opration 
    param inputs: Output from the previous layer(flatten)
    '''
    def call(self, inputs, mask=None):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        # inputs:
        # x: features, y_mask: 1-D or one-hot label works as mask
        x = inputs[0]
        y_mask = inputs[1]
        if y_mask.shape[-1]==1:
            y_mask = K.cast(y_mask, tf.int32)
            y_mask = K.reshape(K.one_hot(y_mask, self.class_num),(-1, self.class_num))

        # feature norm
        x = K.l2_normalize(x, axis=1)
        # weights norm
        self.W = K.l2_normalize(self.W, axis=0)

        # cos(theta+m)
        cos_theta = K.dot(x, self.W)
        cos_theta2 = K.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = K.sqrt(sin_theta2 + K.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - threshold
        cond = K.cast(K.relu(cond_v), dtype=tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)

        # mask by label
        y_mask =+ K.epsilon()
        inv_mask = 1. - y_mask
        s_cos_theta = self.s * cos_theta
        output = K.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * y_mask))

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.class_num

    
class EveryEpochCheckpoint(Callback):
    """
    define new callbacks, in order to save model in every epoch
    """
    def __init__(self, model):
        self.model_to_save=model
        
    def on_epoch_end(self, epoch, logs=None):  #log must be defined
        self.model_to_save.save('model9_11fire/model_%d.h5' % epoch)    
      
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
            self.model_to_save.save('model9_11fire/best_model.h5')

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
    
    batch_size = 200
    epoch = 25
    model = Model_train(batch_size, epoch, input_shape, num_class, path_data)
    
    if not os.path.exists('model9_11fire'):
        os.makedirs('model9_11fire')
        
    print('\nTrain_Starting-------------------------------------------------------------------------------')
    model.Squeezefacenet()  
    model.train(path_data)
    
    print('Model Saved.')
    model.save_model(file_path = 'model9_11fire/last_model.h5')
    
    K.clear_session()
   
#    model = Model_train()
#    print('\nTesting---------------------------------------------------------')
#    file_path = 
#    model.load_model(file_path)
#    model.evaluate(dataset)
   

