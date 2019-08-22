# -*- coding: utf-8 -*-
# /usr/bin/env/python3

import sklearn

#import nets.TinyMobileFaceNet as TinyMobileFaceNet
import nets.SimpleNet_256 as SimpleNet_256
import tensorflow as tf
import argparse
import cv2
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('E:/sign_system/execute_system/MTCNN_Tensorflow_fast')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net

slim = tf.contrib.slim

class mobilefacenet(object):
    def __init__(self):

        with tf.Graph().as_default():
            args = self.get_parser()

            # define placeholder
            self.inputs = tf.placeholder(name='img_inputs',
                                         shape=[None, args.image_size[0], args.image_size[1], 3],
                                         dtype=tf.float32)
            self.phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                                       shape=None,
                                                                       name='phase_train')
            # identity the input, for inference
            inputs = tf.identity(self.inputs, 'input')

            if args.model_type == 0:
                prelogits = SimpleNet_256.inference(images=inputs,
                                                    phase_train=self.phase_train_placeholder,
                                                    weight_decay=args.weight_decay)
                
            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            # define sess
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=args.log_device_mapping,
                                    gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # saver to load pretrained model or save model
            saver = tf.train.Saver(tf.trainable_variables())

            # init all variables
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

            # load pretrained model
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)

#                output_graph_def = tf.GraphDef()
#                pb_path = os.path.join(args.pretrained_model, 'MobileFaceNet_9925_9680.pb')
#
#                with open(pb_path, "rb") as f:
#                    output_graph_def.ParseFromString(f.read())
#                    _ = tf.import_graph_def(output_graph_def, name="")
                               
                checkpoint_path = os.path.join(args.pretrained_model, 'SimpleNet_iter_454000.ckpt')
                saver.restore(self.sess, checkpoint_path)
                
    def get_parser(self):
        parser = argparse.ArgumentParser(description='parameters to train net')
        parser.add_argument('--image_size', default=[96, 96], help='the image size')
        parser.add_argument('--weight_decay', default=5e-6, help='L2 weight regularization.')
        parser.add_argument('--pretrained_model', type=str, default='E:/SimpleNet_256/output/ckpt',
                            help='Load a pretrained model before training starts.')
        parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
        parser.add_argument('--model_type', default=0, help='MobileFaceNet or TinyMobileFaceNet')

        args = parser.parse_args()
        return args

    def get_feature(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        
        feed_dict = {self.inputs: inputs, self.phase_train_placeholder: False}
        feature = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return feature

    def crop_pic_extract(self, path_name):          
        path = r'E:/sign_system/extract'
        dirs = os.listdir(path)
        print(dirs)
        for file in dirs:
            labels = file
            if not labels:
                break
            path_pic =  'E:/sign_system/extract'+'/'+labels
            dirspic = os.listdir(path_pic)
            for i in dirspic:
                image = cv2.imread(os.path.join(path_pic, i))
                if image is None:
                    continue
                
                # Turn the image into an array.
                image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
                image = image.astype('float32')
                image = image - 127.5
                image = image * 0.0078125   
                t1 = time.time()
                f1 = self.get_feature(image)
                t2 = time.time()
                print('time cost:',t2-t1)
                
            f=open(path_name, 'a')
            f.write(labels)
            f.write('\n')
            count = 0
            for element in f1.flat:
                count = count + 1
                f.write(str(element))
                f.write(' ')
                if int(count%10) ==0:
                    f.write('\n')
            f.write('\n')
            f.close()  
                
    def build_camera(self, camera_id, path):
        count = 500
        thresh = [0.9, 0.9, 0.8]
        min_face_size = 100
        stride = 2
        slide_window = False
        detectors = [None, None, None]
        prefix = ['E:/sign_system/execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model_V2/PNet_landmark/PNet', 
                  'E:/sign_system/execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model_V2/RNet_landmark/RNet', 
                  'E:/sign_system/execute_system/MTCNN_Tensorflow_fast/data/MTCNN_model_V2/ONet_landmark/ONet']
        epoch = [40, 36, 36]
        model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
        PNet = FcnDetector(P_Net, model_path[0])
        detectors[0] = PNet
        RNet = Detector(R_Net, 24, 1, model_path[1])
        detectors[1] = RNet
        ONet = Detector(O_Net, 48, 1, model_path[2])
        detectors[2] = ONet        
        mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
        cap = cv2.VideoCapture(camera_id)
        num = 0
        cur = self.read_feature(path)
        while True:
            success, frame = cap.read()
            thickness = (frame.shape[0] + frame.shape[1]) // 350
            if success:
                t1 = time.time() 
                image = np.array(frame)
                boxes_c,landmarks = mtcnn_detector.detect(image)
                #print(boxes_c)
                for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]
                        #score = boxes_c[i, 4]
                        cropbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        W = -int(cropbbox[0]) + int(cropbbox[2])
                        H = -int(cropbbox[1]) + int(cropbbox[3])
                        paddingH = 0.02 * H
                        paddingW = 0.01 * W
                        crop_img = frame[int(cropbbox[1]+paddingH):int(cropbbox[3]-paddingH), 
                                       int(cropbbox[0]-paddingW):int(cropbbox[2]+paddingW)]
                        
                        image = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_CUBIC)  
                                                                
                        image = image.astype('float32')
                        image = image - 127.5
                        image = image * 0.0078125  
                        
                        f1_emb = self.get_feature(image)
                        f1 = f1_emb.reshape(256)
                        #计算距离
                        d1 = 100
                        show_name = ''
                        embed1 = sklearn.preprocessing.normalize(f1_emb)
                        for n,v in cur.items():
                            v = np.array(v)
                            v_emb = v.reshape(1, 256)
                            embed2 = sklearn.preprocessing.normalize(v_emb)
                            diff = np.subtract(embed1, embed2)
                            dist = np.sum(np.square(diff), 1)
                            
                            d=np.dot(v,f1)/(np.linalg.norm(v)*np.linalg.norm(f1))
                            print("name: %s total cosin distance %f and Euclidean distance %f"%(n, d, dist))
                            if dist < d1:
                                d1 = dist
                                show_name = str(n)
                            else:
                                pass
                        #print(show_name)
                        t2 = time.time()
                        delta_t = t2-t1
                        text_start = (max(int(cropbbox[0]), 10), max(int(cropbbox[1]), 10))
                        cv2.putText(frame, show_name, text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1)  
                        cv2.putText(frame, "time cost:" + '%.04f'%delta_t,(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 2)                                                          
                        # rectangle for face area
                        for i in range(thickness):
                            start = (int(cropbbox[0]) + i, int(cropbbox[1]) + i)
                            end = (int(cropbbox[2] - i), int(cropbbox[3]) - i)
                            frame = cv2.rectangle(frame, start, end, (0, 255, 0), 1)  
                            
                        # display the landmarks
                        for i in range(landmarks.shape[0]):
                            for j in range(len(landmarks[i])//2):
                                cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255)) 
                        num = num +1                
                cv2.imshow("Camera", frame)
                k = cv2.waitKey(10)
                # 如果输入q则退出循环
                if (k & 0xFF == ord('q') or count == num):
                    break  
            else:
                print ('device not find')
                break
        cap.release()
        cv2.destroyAllWindows()                     
            
    def read_feature(self,path_name):      
        f = open(path_name,'r')
        cur_entry = {}
        numlist = []
        while True:
            line = f.readline().strip('\n')   
            if not line:
                break
            names = line
            for i in range(26):
                nums = f.readline().strip('\n').split(' ')
                for j in nums:
                    try: 
                        float(j)
                        numlist.append(float(j)) 
                    except ValueError:  
                        pass  
            cur_entry[names] = numlist
            numlist = []
        f.close()
        
        return cur_entry

if __name__ == '__main__':
    t1 = time.time()
    model = mobilefacenet()
    t2 = time.time()
    
    camera_id = 0
    path_name = r'E:/SimpleNet_256/feature.txt'
    if not os.path.exists(path_name):
        ffeature = open(path_name, 'w')
        ffeature.close()
    
    #model.crop_pic_extract(path_name)
    
    model.build_camera(camera_id, path_name)  
    
    img_num = 500
    t3 = time.time()
    print('init model cost {} s, get image feature speed {} fps'.format((t2 - t1), img_num / (t3 - t2)))
