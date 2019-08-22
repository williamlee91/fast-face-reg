# fast-face-reg

some better performance models were uploaded 
I advise you to retrain these models, because some details of these models were improved.

The following is all from my paper(fast face recognition model design without pruning). If you want to know the details, you can contact me.

1) Details

1、channels split fire module（CSF）
2、channels split fire module with seperable convolution(CSFS)
![image](https://github.com/williamlee91/fast-face-reg/blob/master/tensorflow_reg/images/module.bmp)
3、downsampling depth_wise convolution (DDWConv)
![image](https://github.com/williamlee91/fast-face-reg/blob/master/tensorflow_reg/images/downsampling.bmp)
4、Retrain with mini-batch
  paer:Revisiting small batch training for deep neural networks
5、ELU, ReLU, CReLU

2) Models

![image](https://github.com/williamlee91/fast-face-reg/blob/master/tensorflow_reg/images/CSN.bmp)

model 1 ->  CSN 
 The channels after each downsampling operations are 64, 96, 128, 192. In the first convolution layer, the channels are set to 64.

model 3 -> CSN-Fast
  I modify the CSN-fast to substitute four CSF and two Dconv for CSFS and EDDWConv in CSN while maintaining network structure basically unchanged. 

model 5 -> CSN-Faster
  In CSF-faster, there are 7 CSF modules, 2 DConv modules and 2 EDDWConv modules in CSF-faster

model 9_4 ->CSN-Fastest
  In order to observe cost of time, we set the profiler in FPS testing. I found that the shallower the layer, the more time-consuming it is. So i set CReLu in shallow layers, and modify the number of channels in shallow and middle layers.

3)Results

  The best 
![image](https://github.com/williamlee91/fast-face-reg/blob/master/tensorflow_reg/images/results.png)

