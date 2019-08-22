# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:06:12 2019
@author: william
"""

"""
1、定义标签编码。在输入训练数据中，标签用单词表示，但我们需要数字来训练系统。
2、从每幅图中提取ROI属性值和标签编码器。
3、加载人脸级联文件。
4、生成局部二值模式直方图人脸识别器。
5、用训练集数据训练人脸识别器。
6、加载测试数据所在路径，将批量数据读取到内存，用人脸级联文件来检测器确定人脸的位置。
7、对于每个人的脸ROI，运行人脸识别器，将标签转换为单词。
8、直接在图片打上分类文字，并展示。
9、关闭cv2。
--------------------- 
参考原文：https://blog.csdn.net/weixin_42039090/article/details/80684886 

"""
import os
import cv2
import numpy as np
import time
from sklearn import preprocessing

          
# Class to handle tasks related to label encoding
class LabelEncoder(object):
   
    # Method to encode labels from words to numbers
    def encode_labels(self, label_words):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(label_words)
    
    # Convert input label from word to number
    def word_to_num(self, label_word):
        #print("[label_word]",[label_word])
        #print("self.le.transform([label_word])[0]",self.le.transform([label_word])[0])
        #print("int(self.le.transform([label_word])[0])",int(self.le.transform([label_word])[0]))
        return int(self.le.transform([label_word])[0])
    
    # Convert input label from number to word
    def num_to_word(self, label_num):
        return self.le.inverse_transform([label_num])[0]
   
# Extract images and labels from input path
def get_images_and_labels(faceCascade,input_path):
    label_words = []
    # Iterate through the input path and append files
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            label_words.append(filepath.split('\\')[-2])
    #print("label_words",label_words)
    #print("filepath",filepath)
    # Initialize variables
    images = []
    
    le = LabelEncoder()
    le.encode_labels(label_words)
    #print("le",le)
    labels = []
    # Parse the input directory
    for root, dirs, files in os.walk(input_path):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            # Read the image in grayscale format
            image = cv2.imread(filepath, 0)
            # Extract the label
            name = filepath.split('\\')[-2]
            # Perform face detection
            faces = faceCascade.detectMultiScale(image, 1.1, 2, minSize=(100,100))
            # Iterate through face rectangles
            # print(faces)
            for (x, y, w, h) in faces:
                images.append(image[y:y+h, x:x+w])
                labels.append(le.word_to_num(name))
                #print("x,y,w,h",x,y,w,h,"filepath=",filepath)
    #print("labels=",labels)
    return images, labels, le

class my_face_reconginizer:
    
    def __init__(self,cascade_path = r"E:\sign_system\opencv\haarcascades\haarcascade_frontalface_alt2.xml",
                 path_train = r'E:\sign_system\face_data_for32',
                 path_test = r'E:\sign_system\face_test'):
        #人脸级联文件
        self.faceCascade = cv2.CascadeClassifier(cascade_path) 
        self.path_train=path_train
        self.path_test=path_test
        self.images, self.labels, self.le = get_images_and_labels(self.faceCascade,self.path_train)
        
    def Recongizer_train(self):

        # 创建识别模型，使用EigenFace算法识别，Confidence评分低于4000是可靠
        # self.recognizer = cv2.face.EigenFaceRecognizer_create()
        # 创建识别模型，使用LBPHFace算法识别，Confidence评分低于50是可靠
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # 创建识别模型，使用FisherFace算法识别，Confidence评分低于4000是可靠
        #self.recognizer = cv2.face.FisherFaceRecognizer_create()

        # 训练模型
        # train函数参数：images, labels，两参数必须为np.array格式，而且labels的值必须为整型
        print( "\nTraining...")
        self.recognizer.train(self.images, np.array(self.labels))
        
        self.recognizer.save(r'E:\sign_system\execute_system\trainner\trainner.yml')
    
    def Recongizer_Predict(self):
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # recognizer = cv2.createLBPHFaceRecognizer() # in OpenCV 2
        self.recognizer.read('trainner/trainner.yml')
        # recognizer.load('trainner/trainner.yml') # in OpenCV 2
                
        # Test the recognizer on unknown images
        print('\nPerforming prediction on test images...')
        stop_flag = False
        for root, dirs, files in os.walk(self.path_test):
            for filename in (x for x in files if x.endswith('.jpg')):
                filepath = os.path.join(root, filename)
                # Read the image
                predict_image = cv2.imread(filepath, 0)
                print("predict_image",predict_image)
                #print("filepath",filepath)
                #print("type(predict_image)",type(predict_image))
                #print("shape(predict_image)",np.shape(predict_image))
                # Detect faces
                faces = self.faceCascade.detectMultiScale(predict_image, 1.1,2, minSize=(100,100))
                print("faces",faces)
                # Iterate through face rectangles
                for (x, y, w, h) in faces:
                    # Predict the output
                    predicted_index, conf = self.recognizer.predict(
                                predict_image[y:y+h, x:x+w])
                    # Convert to word label
                    predicted_person = self.le.num_to_word(predicted_index)
                    # Overlay text on the output image and display it
                    cv2.putText(predict_image, 'Prediction: ' + predicted_person,
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
                    cv2.imshow("Recognizing face", predict_image)
                                        
                c = cv2.waitKey(5)
                if c == 27:
                    stop_flag = True
                    break
            if stop_flag:
                break
    def detect_test(self):
        
        color = (255, 0, 0)
        
        cv2.namedWindow("Image1")  
        impath1 = r'E:\opencvtestface\0_Parade_marchingband_1_60.jpg'        
        predict_image1 = cv2.imread(impath1, 0)
        t = time.time()
        faces1 = self.faceCascade.detectMultiScale(predict_image1, 1.1,2, minSize = (5, 5))
        t1 = time.time() - t
        print (t1) 
        count = len(faces1)
        for (x1, y1, w1, h1) in faces1:
            frame = cv2.rectangle(predict_image1, (x1, y1), (x1 + w1, y1 + h1), color, thickness = 2)  
            cv2.imshow("Image1", frame)
        cv2.putText(predict_image1, "detected faces:"+str(count),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 255), 2)
            #cv2.imwrite("result_landmark/%d.png" %(count),image)
            #cv2.imshow("pic",image)
        res_path = r'E:\opencvtestface\result1.jpg'
        cv2.imwrite(res_path, predict_image1)
        cv2.waitKey (0)  
        cv2.destroyAllWindows() 
         
        impath2 = r'E:\opencvtestface\timg.jpg'       
        predict_image2 = cv2.imread(impath2, 0)
        t = time.time()
        faces2 = self.faceCascade.detectMultiScale(predict_image2, 1.1,2, minSize = (5, 5))
        t2 = time.time() - t
        print (t2)        
        count2 = len(faces2)
        for (x1, y1, w1, h1) in faces2:
            frame = cv2.rectangle(predict_image2, (x1, y1), (x1 + w1, y1 + h1), color, thickness = 2)  
            cv2.imshow("Image1", frame)
        cv2.putText(predict_image2, "detected faces:"+str(count2),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
        #cv2.imwrite("result_landmark/%d.png" %(count),image)
        #cv2.imshow("pic",image)
        res_path = r"E:\opencvtestface\result2.jpg"
        cv2.imwrite(res_path, predict_image2)
        cv2.waitKey (0)  
        cv2.destroyAllWindows() 
 
    def face_predict(self):       
        # 开启摄像头
        camera = cv2.VideoCapture(0)
        # 加载Haar级联数据文件，用于检测人面
        #self.face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # recognizer = cv2.createLBPHFaceRecognizer() # in OpenCV 2
        self.recognizer.read('trainner/trainner.yml')
        # recognizer.load('trainner/trainner.yml') # in OpenCV 2

        while (True):
            # 检测摄像头的人面
            read, img = camera.read()
            faces = self.faceCascade.detectMultiScale(img, 1.3, 5,minSize = (16, 16))
            # 将检测的人面进行识别处理
            for (x, y, w, h) in faces:
                
                
                # 画出人面所在位置并灰度处理
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray[x:x + w, y:y + h]
        
                # 将检测的人面缩放200*200大小，用于识别
                # cv2.INTER_LINEAR是图片变换方式，其余变换方式如下：
                # INTER_NN - 最近邻插值。
                # INTER_LINEAR - 双线性插值(缺省使用)
                # INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。
                # INTER_CUBIC - 立方插值。
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        
                # 检测的人面与模型进行匹配识别
                predicted_index, conf = self.recognizer.predict(roi)
                print("Label: %s, Confidence: %.2f" % (predicted_index, conf))
                # 将识别结果显示在摄像头上
                # cv2.FONT_HERSHEY_SIMPLEX 定义字体
                # cv2.putText参数含义：图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                # 如果要输出中文字，可参考https://blog.csdn.net/m0_37606112/article/details/78511381
                predicted_person = self.le.num_to_word(predicted_index)
                cv2.putText(img, 'Prediction: ' + predicted_person, 
                            (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
            cv2.imshow("camera", img)
            if cv2.waitKey(120) & 0xff == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("1、训练")
    print("2、test")
    print("3、摄像头")
    print("4、检测")
    test=my_face_reconginizer()
    while True:
        s = input("请输入数字：")
        if(s == '1'):
            test.Recongizer_train()
        elif(s == '2'):
            test.Recongizer_Predict()
        elif(s == '3'):
            test.face_predict()
        elif(s == '4'):
            test.detect_test()
        else:
            break