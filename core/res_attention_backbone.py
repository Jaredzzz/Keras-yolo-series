from attention_block import _darknet_conv, _csp_darknet_block,_residual_attention_block
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Lambda, MaxPooling2D, Input
import os
import keras
from core.activation import Mish, Mish6
from keras.layers.merge import add,concatenate
from keras.models import Model
from keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def residual_attention_csp_darknet_model(input_image):
    # x:32倍降采样输出_大目标
    # route_1：8倍降采样输出_小目标
    # route_2：16倍降采样输出_中目标
    x = _darknet_conv(input_image, {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                    'activation': 'mish'})
    x = _darknet_conv(x, {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False,
                          'activation': 'mish'})

    x = _residual_attention_block(input=x,
                                  convs=[
                                      {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'}],
                                  num_blocks=1,
                                  encoder_depth=3)
    x = _darknet_conv(x, {'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False,
                          'activation': 'mish'})

    x = _residual_attention_block(input=x,
                                  convs=[
                                      {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'}],
                                  num_blocks=2,
                                  encoder_depth=2)
    x = _darknet_conv(x, {'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False,
                          'activation': 'mish'})

    x = _residual_attention_block(input=x,
                                  convs=[
                                      {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'}],
                                  num_blocks=8,
                                  encoder_depth=1)
    route_1 = x
    x = _darknet_conv(x, {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False,
                          'activation': 'mish'})
    x = _residual_attention_block(input=x,
                                  convs=[
                                      {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'}],
                                  num_blocks=8,
                                  encoder_depth=1)
    route_2 = x
    x = _darknet_conv(x, {'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'sparse_factor': False,
                          'activation': 'mish'})

    x = _residual_attention_block(input=x,
                                  convs=[
                                      {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'},
                                      {'filter': 1024, 'kernel': 1, 'stride': 1, 'bnorm': True, 'sparse_factor': False,
                                       'activation': 'mish'}],
                                  num_blocks=4,
                                  encoder_depth=1)

    return route_1, route_2, x


# input_image = Input(shape=(512, 512, 3))
# x = residual_attention_csp_darknet_model(input_image)
# model = Model(inputs=input_image, outputs=x, name='residual_attention_csp_darknet53_model')
# model.summary()
# plot_model(model, "residual_attention_csp_darknet_model.png", show_shapes=True, show_layer_names=True)
