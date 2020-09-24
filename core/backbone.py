from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D
from keras.layers.merge import add,concatenate
from keras.models import Model
from keras.initializers import glorot_uniform
import os
from keras.utils import plot_model
from core.activation import Mish
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES'] ="0"
'''
darknet53 model 
'''
def _darknet_conv(input,conv):
    x = input
    if conv['stride'] == 1 :
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


def _darknet_conv_block(input,convs):
    x = input
    for conv in convs:
        x = _darknet_conv(input=x,conv=conv)
    return x


def _darknet_res_block(input,convs,num_blocks):
    x = input
    x = _darknet_conv(x,convs[0])
    ori_layer_idx_1,ori_layer_idx_2 = convs[1]['layer_idx'],convs[2]['layer_idx']
    for i in range(num_blocks):
        update_layer_idx_1,update_layer_idx_2 = ori_layer_idx_1 + i*3,ori_layer_idx_2 + i*3
        convs[1].update({'layer_idx':update_layer_idx_1})
        convs[2].update({'layer_idx': update_layer_idx_2})
        y = _darknet_conv(x,convs[1])
        y = _darknet_conv(y,convs[2])
        x = add([x, y])
    return x


def darknet53_model(input_image):
    # x:32倍降采样输出_大目标
    # route_1：8倍降采样输出_中目标
    # route_2：16倍降采样输出_小目标
    # layer:0
    x = _darknet_conv(input_image,{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 0})

    # layer:1-4
    x = _darknet_res_block(x,[{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 1},
                              {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 2},
                              {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 3}],
                           num_blocks=1)
    # layer:5-11
    x = _darknet_res_block(x,[{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 5},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 6},
                              {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 7}],
                           num_blocks=2)
    # layer:12-36
    x = _darknet_res_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 12},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 13},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 14}],
                           num_blocks=8)
    route_1 = x
    # layer:37-61
    x = _darknet_res_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 37},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 38},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 39}],
                           num_blocks=8)
    route_2 = x
    # layer:62-74
    x = _darknet_res_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 62},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 63},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'leaky', 'layer_idx': 64}],
                           num_blocks=4)
    return route_1, route_2, x


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


def csp_darknet53_model(input_image):
    # layer:0
    x = _darknet_conv(input_image,{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 0})
    # layer:1-9
    x = _csp_darknet_block(x,[{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'mish', 'layer_idx': 1},
                            {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 2},
                            {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 5},
                            {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 6},
                            {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 10}],
                           num_blocks=1)
    # layer:10-21
    x = _csp_darknet_block(x,[{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'mish', 'layer_idx': 11},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 12},
                              {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 15},
                              {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 16},
                              {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 23}],
                           num_blocks=2)
    # layer:22-51
    x = _csp_darknet_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'mish', 'layer_idx': 24},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 25},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 28},
                               {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 29},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 54}],
                           num_blocks=8)
    route_1 = x
    # layer:52-81
    x = _csp_darknet_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'mish', 'layer_idx': 55},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 56},
                               {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 59},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 60},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 85}],
                           num_blocks=8)
    route_2 = x
    # layer:82-99
    x = _csp_darknet_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activation': 'mish', 'layer_idx': 86},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 87},
                               {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 90},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 91},
                               {'filter': 1024, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activation': 'mish', 'layer_idx': 104}],
                           num_blocks=4)

    return route_1, route_2, x

# input_image = Input(shape=(None,None,3))
# _,_,x = csp_darknet53_model(input_image)
# model = Model(inputs=input_image, outputs=x, name='darknet')
# # model.summary()
# plot_model(model,"model.png",show_shapes=True,show_layer_names=True)