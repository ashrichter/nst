# -*- coding: utf-8 -*-

import scipy.io
import tensorflow as tf

# for mask
vgg_layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

'''
Usage:
    only use vgg_structure().build(input_image)
    
    Parameters: input_image: nparray
    
    e.g.
    After:
    structure = vgg_structure()
    structure.build(input_image)
    
    structure.conv1_1 will give the output (a tensorflow variable) of layer "conv1_1"
'''
def file_vgg_read(pretrained_vgg):
    
    mat = scipy.io.loadmat(pretrained_vgg)
    layers_info = mat["layers"]
    # normalization_info = mat["normalization"]
    
    return layers_info #, normalization_info


def conv2d_relu(layers_info, prev_layer, layer_id):
    
    W = layers_info[0][layer_id][0][0][2][0][0]
    b = layers_info[0][layer_id][0][0][2][0][1]
    b = b.reshape(b.size)
    
    #W = tf.Variable(W, trainable = False)
    #b = tf.Variable(b, trainable = False)
    
    W = tf.constant(W)
    b = tf.constant(b)
    
    
    conv2d = tf.nn.conv2d(input=prev_layer, \
                          filters=W, \
                          strides=[1, 1, 1, 1], \
                          padding="SAME")
    output = tf.nn.relu(conv2d + b)
    return output

def pool(prev_layer):
    
    output = tf.compat.v1.nn.avg_pool(value=prev_layer, \
                            ksize=[1, 2, 2, 1], \
                            strides=[1, 2, 2, 1], \
                            padding="SAME")
    
    # if max pooling, use this output:****************************************************
    '''
    output = tf.nn.max_pool(value=prev_layer, \                                 
                            ksize=[1, 2, 2, 1], \                                 
                            strides=[1, 2, 2, 1], \                                 
                            padding="SAME")
    '''

    return output


class vgg_structure():
    
    def build(self, input_image, layers_info):
        #layers_info = file_vgg_read("imagenet-vgg-verydeep-19.mat")
        
        # build the vgg training net
        input_layer = input_image
        
        setattr(self, "conv1_1" , conv2d_relu(layers_info, prev_layer = input_layer, layer_id = 0) )
        setattr(self, "conv1_2" , conv2d_relu(layers_info, prev_layer = self.conv1_1, layer_id = 2) )
        setattr(self, "pool_1" , pool(prev_layer = self.conv1_2) )
        
        setattr(self, "conv2_1" , conv2d_relu(layers_info, prev_layer = self.pool_1, layer_id = 5) )
        setattr(self, "conv2_2" , conv2d_relu(layers_info, prev_layer = self.conv2_1, layer_id = 7) )
        setattr(self, "pool_2" , pool(prev_layer = self.conv2_2) )
        
        setattr(self, "conv3_1" , conv2d_relu(layers_info, prev_layer = self.pool_2, layer_id = 10) )
        setattr(self, "conv3_2" , conv2d_relu(layers_info, prev_layer = self.conv3_1, layer_id = 12) )
        setattr(self, "conv3_3" , conv2d_relu(layers_info, prev_layer = self.conv3_2, layer_id = 14) )
        setattr(self, "conv3_4" , conv2d_relu(layers_info, prev_layer = self.conv3_3, layer_id = 16) )
        setattr(self, "pool_3" , pool(prev_layer = self.conv3_4) )
        
        setattr(self, "conv4_1" , conv2d_relu(layers_info, prev_layer = self.pool_3, layer_id = 19) )
        setattr(self, "conv4_2" , conv2d_relu(layers_info, prev_layer = self.conv4_1, layer_id = 21) )
        setattr(self, "conv4_3" , conv2d_relu(layers_info, prev_layer = self.conv4_2, layer_id = 23) )
        setattr(self, "conv4_4" , conv2d_relu(layers_info, prev_layer = self.conv4_3, layer_id = 25) )
        setattr(self, "pool_4" , pool(prev_layer = self.conv4_4) )
        
        setattr(self, "conv5_1" , conv2d_relu(layers_info, prev_layer = self.pool_4, layer_id = 28) )
        setattr(self, "conv5_2" , conv2d_relu(layers_info, prev_layer = self.conv5_1, layer_id = 30) )
        setattr(self, "conv5_3" , conv2d_relu(layers_info, prev_layer = self.conv5_2, layer_id = 32) )
        setattr(self, "conv5_4" , conv2d_relu(layers_info, prev_layer = self.conv5_3, layer_id = 34) )
        setattr(self, "pool_5" , pool(prev_layer = self.conv5_4) )
        
    def buildmask(self, input_mask, mask_downsample_type):
        net = {}
        current = input_mask

        # soft
        if mask_downsample_type == 'simple':
            for name in vgg_layers:
                layer_kind = name[:4]
                if layer_kind == 'pool':
                    current = tf.nn.avg_pool2d(input=current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                setattr(self, name, current)
        # hard
        elif mask_downsample_type == 'all':
            for name in vgg_layers:
                layer_kind = name[:4]
                if layer_kind == 'conv':
                    current = tf.nn.max_pool2d(input=current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                elif layer_kind == 'pool':
                    current = tf.nn.max_pool2d(input=current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                setattr(self, name, current)     
        # hard, keep the padding boundary unchanged
        elif mask_downsample_type == 'inside':
            current = 1 - current
            for name in vgg_layers:
                layer_kind = name[:4]
                if layer_kind == 'conv':
                    current = tf.nn.max_pool2d(input=current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                elif layer_kind == 'pool':
                    current = tf.nn.max_pool2d(input=current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                setattr(self, name, current) 
        # soft
        elif mask_downsample_type == 'mean':
            for name in vgg_layers:
                layer_kind = name[:4]
                if layer_kind == 'conv':
                    current = tf.nn.avg_pool2d(input=current, ksize=[1,3,3,1], strides=[1,1,1,1], padding='SAME')
                elif layer_kind == 'pool':
                    current = tf.nn.avg_pool2d(input=current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                setattr(self, name, current)  

