from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Lambda, MaxPooling2D, Input
import os
import keras
from core.activation import Mish, Mish6
from keras.layers.merge import add,concatenate
from keras.models import Model
from keras.utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
darknet tools
'''


def _darknet_conv(input, conv):
    x = input
    if conv['stride'] == 1:
        padding = "same"
    else:
        padding = "valid"

    if conv['stride'] > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # unlike tensorflow darknet prefer left and top paddings
    x = Conv2D(conv['filter'],
               conv['kernel'],
               strides=conv['stride'],
               padding=padding,
               # unlike tensorflow darknet prefer left and top paddings
               name='conv_' + str(conv['layer_idx']),
               use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']:
        x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['activation'] == "leaky":
        x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
    elif conv['activation'] == "mish":
        x = Mish(name='mish_' + str(conv['layer_idx']))(x)
    else:
        pass
    return x


def _darknet_conv_block(input, convs):
    x = input
    for conv in convs:
        x = _darknet_conv(input=x, conv=conv)
    return x


def slice_channel(x, index):
    return x[:, :, :, 0: index]


def _csp_darknet_tiny_res_block(input, convs):
    x = input
    x = _darknet_conv(x, convs[0])
    channel = keras.backend.int_shape(x)[-1]
    x1 = Lambda(slice_channel, arguments={'index': int(channel/2)})(x)
    x2 = _darknet_conv(x1, convs[1])
    x3 = _darknet_conv(x2, convs[2])
    route = concatenate([x2, x3])
    x4 = _darknet_conv(route, convs[3])
    y = concatenate([x, x4])
    return y, x4


def _csp_darknet_block(input,convs,num_blocks):
    x = input
    x = _darknet_conv(x,convs[0])
    x2 = _darknet_conv(x,convs[1])
    update_layer_idx_1 = convs[1]['layer_idx']+2
    convs[1].update({'layer_idx': update_layer_idx_1})
    x1 = _darknet_conv(x,convs[1])
    ori_layer_idx_1,ori_layer_idx_2 = convs[2]['layer_idx'],convs[3]['layer_idx']
    for i in range(num_blocks):
        update_layer_idx_2,update_layer_idx_3 = ori_layer_idx_1 + i*3,ori_layer_idx_2 + i*3
        convs[2].update({'layer_idx':update_layer_idx_2})
        convs[3].update({'layer_idx': update_layer_idx_3})
        y = _darknet_conv(x1,convs[2])
        y = _darknet_conv(y,convs[3])
        x1 = add([x1, y])
    update_layer_idx_1 = convs[1]['layer_idx']+num_blocks*3+1
    convs[1].update({'layer_idx': update_layer_idx_1})
    x1 = _darknet_conv(x1, convs[1])
    x = concatenate([x1,x2])
    x = _darknet_conv(x, convs[4])
    return x


def yolov4_tiny_backbone(input_image):

    # stage1:layer:0-1(conv)
    x = _darknet_conv_block(input=input_image, convs=[
        {'filter': 32, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 0},
        {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 1}])

    # stage2:layer:2-6(conv, maxpool)
    x, _ = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 2},
        {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 3},
        {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 4},
        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 5}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # stage3:layer:7-11(conv, maxpool)
    x, _ = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 7},
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 8},
        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 9},
        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 10}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # stage3:layer:12-16(conv, maxpool)
    x, route = _csp_darknet_tiny_res_block(x, convs=[
        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 12},
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 13},
        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 14},
        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 15}])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    return route, x


def csp_darknet53_model(input_image):
    # x:32倍降采样输出_大目标
    # route_1：8倍降采样输出_小目标
    # route_2：16倍降采样输出_中目标
    # layer:0
    x = _darknet_conv(input_image,{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 0})
    # layer:1-9
    x = _csp_darknet_block(x,[{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 1},
                            {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 2},
                            {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 5},
                            {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 6},
                            {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 10}],
                           num_blocks=1)
    # layer:10-21
    x = _csp_darknet_block(x,[{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 11},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 12},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 15},
                              {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 16},
                              {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 23}],
                           num_blocks=2)
    # layer:22-51
    x = _csp_darknet_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 24},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 25},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 28},
                               {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 29},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 54}],
                           num_blocks=8)
    route_1 = x
    # layer:52-81
    x = _csp_darknet_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 55},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 56},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 59},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 60},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 85}],
                           num_blocks=8)
    route_2 = x
    # layer:82-99
    x = _csp_darknet_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 86},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 87},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 90},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 91},
                               {'filter': 1024, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False, 'activation': 'mish', 'layer_idx': 104}],
                           num_blocks=4)

    return route_1, route_2, x




