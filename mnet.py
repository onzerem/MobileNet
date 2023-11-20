import tensorflow as tf

from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, GlobalAvgPool2D, Flatten, Dense
from tensorflow.keras import Model

# more comments, github repo, mobile ssd
# Depthwise convolution
def depth_block(x, strides):
    x = DepthwiseConv2D(kernel_size = 3, strides=strides, padding='same',  use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Pointwise convolution
def single_conv_block(x,filters):
    x = Conv2D(filters, kernel_size = 1, strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Depthwise separable convolution (depthwise + pointwise)
def combo_layer(x,filters, strides):
    x = depth_block(x,strides)
    x = single_conv_block(x, filters)
    return x

def MobileNet(input_shape=(224,224,3),n_classes = 1000):
    input = Input(input_shape)
    # changing stride from (2,2) to (1,1) so mobilenet can better read input
    # mobilenet made for (224,224,3) images which are much bigger than cifar10 images(32,32,3) 
    x = Conv2D(32,3,strides=(1,1),padding = 'same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = combo_layer(x,64, strides=(1,1))  # stride 1
    x = combo_layer(x,128,strides=(2,2))  # stride 2 
    x = combo_layer(x,128,strides=(1,1))  # stride 1
    x = combo_layer(x,256,strides=(2,2))  # stride 2
    x = combo_layer(x,256,strides=(1,1))  # stride 1
    x = combo_layer(x,512,strides=(2,2))  # stride 2
    
    for _ in range(5):
      x = combo_layer(x,512,strides=(1,1)) # stride 1 x 5
      
    x = combo_layer(x,1024,strides=(2,2))  # stride 2
    x = combo_layer(x,1024,strides=(1,1))  # stride 1
    
    x = GlobalAvgPool2D()(x)
    
    output = Dense(n_classes,activation='softmax')(x)
    model = Model(input, output)
    
    return model


