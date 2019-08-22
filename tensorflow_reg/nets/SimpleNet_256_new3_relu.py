# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

def csnet(images, bottleneck_layer_size, is_training):
    """
    Args:
    images: a tensor of shape [batch_size, height, width, channels].
    embedding_size: number of predicted classes. 
    is_training: whether is training or not.   
      Returns:
    net: a 2D Tensor with the logits
    """
    
    #stage 0
    x0 = slim.conv2d(images, 64, [3,3], stride=2,
                     padding="SAME", scope="conv_dw0")
    
    #stage 1
    x1 = csf(x0, 32, "fire1")  
    x1 = slim.conv2d(x1, 96, [3,3], stride=2,
                     padding="SAME", scope="conv_dw1")
    
    #stage 2
    x2 = csf(x1, 48, "fire2")      
 
    #stage 3
    x3 = csf(x2, 48, "fire3")  
    
    #stage 4
    x4 = csf(x3, 48, "fire4")               
    x4 = ddw_conv(x4, depth=128, scope="dw_conv4")
    
    #stage 5
    x5 = csfs(x4, 64, "fire5")               

    #stage 6
    x6 = csfs(x5, 64, "fire6")            
    
    #stage 7
    x7 = csfs(x6, 64, "fire7")  

    #stage 8
    x8 = csfs(x7, 64, "fire8")             
    x8 = ddw_conv(x8, depth=192, scope="dw_conv8")
            
    #stage 9
    x9 = csfs(x8, 96, "fire9")

    #GDC
    kernel_size = _reduced_kernel_size_for_small_input(x9, [6, 6])
    x9 = slim.separable_conv2d(x9, num_outputs=None, kernel_size=kernel_size, stride=1,
                                padding = "VALID",activation_fn=None,depth_multiplier=1.0, 
                                scope="GDC")
    
    #linear
    x9 = slim.conv2d(x9, 128, [1, 1], stride=1, padding="SAME", activation_fn=None, scope="linear")
    
    # squeeze the axis
    out = tf.squeeze(x9, axis=[1, 2], name="logits")
        
    return out

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out

def ddw_conv(input_tensor, depth, stride=2, scope=None):
    """
    This function is defined as a downsamoling method by depthwise convolution
    :param input_tensor: Input tensor in (N, H, W, in_channels)
    :param depth: channels for the depthwise convolution
    :param stride: stride for downsampling
    :return: Output tensor (B, H_new, W_new, out_channels)
    """
    with tf.variable_scope(scope):
        ds_dwconv_3x3= slim.separable_conv2d(inputs=input_tensor, 
                                          num_outputs=None, 
                                          kernel_size=[3, 3], 
                                          stride=stride, 
                                          activation_fn=None,
                                          padding='SAME',
                                          depth_multiplier=1.0)
        
        ds_conv_1x1 = slim.conv2d(ds_dwconv_3x3, depth, [1, 1],
                                  stride=1, padding="SAME", scope="ds_conv_1x1")
        return ds_conv_1x1

def fire(input_tensor, squeeze_depth, scope=None):
    """
    Normal fire_squeeze block performs conv+bn+relu6 operations.
    :param inputs: Input tensor in (B, H, W, C_in)
    :param out_channels: number of filters in the convolution layer
    :param scope: as its name tells
    :return: Output tensor in (B, H_new, W_new, 8*out_channels)
    """
    with tf.variable_scope(scope):
        squeeze = slim.conv2d(input_tensor, squeeze_depth, [1, 1],
                              stride=1, padding="SAME", scope="squeeze")
        # squeeze
        expand_1x1 = slim.conv2d(squeeze, (4*squeeze_depth), [1, 1],
                                 stride=1, padding="SAME", scope="expand_1x1")
        expand_3x3 = slim.separable_conv2d(inputs=squeeze, 
                                      num_outputs=4*squeeze_depth,
                                      kernel_size=[3, 3], 
                                      stride=1, 
                                      padding="SAME",
                                      activation_fn=None,
                                      depth_multiplier=1.0,
                                      scope = "dwconv_3x3"
                                      )
        return tf.concat([expand_1x1, expand_3x3], axis=3)


def channel_split(inputs, num_splits=2):
    c = inputs.get_shape()[3]
    input1, input2 = tf.split(inputs, [int(c//num_splits), int(c//num_splits)], axis=3, name='split')
    return input1, input2

def csfs(input_tensor, depth, scope=None):
    """
    channels split fire module with seperable convolution
    :param input_tensor: Input tensor in (N, H, W, in_channels)
    :param depth: channels for the depthwise convolution
    :param stride: stride for downsampling
    :return: Output tensor (B, H_new, W_new, out_channels)
    """
    with tf.variable_scope(scope):
        # squeeze
        net_left, net_right = channel_split(input_tensor)
        
        dw_3x3 = slim.separable_conv2d(inputs=net_right, 
                                      num_outputs=None, 
                                      kernel_size=[3, 3], 
                                      stride=1, 
                                      padding="SAME",
                                      activation_fn=None,
                                      depth_multiplier=1.0,
                                      scope = "dwconv_3x3"
                                      )
        
        conv_1x1_right = slim.conv2d(dw_3x3, depth, [1, 1],
                                 stride=1, padding="SAME", scope="conv_1x1_right")
        
        concate = tf.concat([net_left, conv_1x1_right], axis=3, name="caoncat")
        
        conv_1x1 = slim.conv2d(concate, (2*depth), [1, 1],
                                 stride=1, padding="SAME", scope="conv_1x1")
        return conv_1x1

def csf(input_tensor, depth, scope=None):
    """
    channels split fire module with seperable convolution
    :param input_tensor: Input tensor in (N, H, W, in_channels)
    :param depth: channels for the depthwise convolution
    :param stride: stride for downsampling
    :return: Output tensor (B, H_new, W_new, out_channels)
    """
    with tf.variable_scope(scope):
        # squeeze
        net_left, net_right = channel_split(input_tensor)
        
        conv_3x3 = slim.conv2d(net_right, depth, [3,3], stride=1,
                             padding="SAME", scope="conv_3x3")
         
        concate = tf.concat([net_left, conv_3x3], axis=3, name="caoncat")
        
        conv_1x1 = slim.conv2d(concate, (2*depth), [1, 1],
                                 stride=1, padding="SAME", scope="conv_1x1")
        return conv_1x1

def csnet_arg_scope(is_training=True,
                           weight_decay=0.00005,
                           regularize_depthwise=False):
  """Defines the default squeezenet arg scope.
  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
  Returns:
    An `arg_scope` to use for the mobilenet v2 model.
  """
  batch_norm_params = {
      'is_training': is_training,
      'center': True,
      'scale': True,
      'fused': True,
      'decay': 0.995,
      'epsilon': 2e-5,
      # force in-place updates of mean and variance estimates
      'updates_collections': None,
      # Moving averages ends up in the trainable variables collection
      'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES ],
  }

  weights_init = tf.contrib.layers.xavier_initializer(uniform=False)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm):  # tf.keras.layers.PReLU
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer) :
          with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
              return sc

def inference(images, bottleneck_layer_size=128, phase_train=False,
              weight_decay=0.00005, reuse=False):
    '''build a mobilenet_v2 graph to training or inference.
    Args:
        images: a tensor of shape [batch_size, height, width, channels].
        bottleneck_layer_size: number of predicted classes.
        phase_train: Whether or not we're training the model.
        weight_decay: The weight decay to use for regularizing the model.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
    Returns:
        net: a 2D Tensor with the logits 

    '''
    # pdb.set_trace()
    arg_scope = csnet_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        return csnet(images, bottleneck_layer_size=bottleneck_layer_size, is_training=phase_train)