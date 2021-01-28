import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import sigmoid, tanh,  relu ,softmax
from tensorflow.keras.layers import Conv1D, Input ,Dot ,Add, Multiply, Flatten
from tensorflow.keras import Model

def gate(x):
    res_sigmoid = sigmoid(x)
    res_tanh = tanh(x)
    res = Multiply()([res_sigmoid, res_tanh])
    return res
    

def causual_conv(filters, kernel_size , input_shape):
    return Conv1D(filters, kernel_size,input_shape=input_shape,padding='causal')
    
    
def dilated_conv(filters, kernel_size,rate):
    return Conv1D(filters, kernel_size,dilation_rate = rate,padding='same')


def conv_1D(filters, kernel_size):
    # 1X1 conv kernel_size = 1
    return Conv1D(filters, kernel_size,padding='same')



def residual_block(x,filters,kernel_size,rate):

    x = dilated_conv(filters,kernel_size,rate)(x)
    x = gate(x)
    x =  conv_1D(256,1)(x)
    
    return x
        

# 16 我電腦的極限
def generate_model(input_shape, residul_blocks = 16,residul_stacks = 2):
    inputs = Input(shape=input_shape)
    x = causual_conv(256, 3 ,input_shape)(inputs)
    
    rate = 0
    layers_per_stack = residul_blocks // residul_stacks
    
    for k in range (residul_blocks+1):
        rate = 2**(k % layers_per_stack)
        residul_x = x
        x = residual_block(x,256,3,rate)
        # 一開始
        if k == 0:
            skip_x = x
        else:
            skip_x = Add()([skip_x,x])
        
        x = Add()([x,residul_x])
        if k == residul_blocks:
            x = skip_x
            break


    x =  relu(x)
    x =  conv_1D(256,1)(x)
    x =  relu(x)
    x =  conv_1D(256,1)(x)
    #x =  softmax(x)
    
    
    model = Model(inputs=inputs, outputs=x,name="my_wavenet")
    
    return model
    
    
