from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Activation, MaxPool2D,ZeroPadding2D, LeakyReLU
from core.activation import Mish, Mish6
from keras.layers import Add
from keras.layers import Multiply
from keras.layers import Lambda
from keras.layers.merge import add,concatenate


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
               # unlike tensorflow darknet prefer left and top paddings),
               use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']:
        x = BatchNormalization(epsilon=0.001)(x)
    if conv['activation'] == "leaky":
        x = LeakyReLU(alpha=0.1)(x)
    elif conv['activation'] == "mish":
        x = Mish()(x)
    else:
        pass
    return x


def _csp_darknet_block(input=None, convs=None, num_blocks=None):
    x = input
    x2 = _darknet_conv(x,convs[0])
    x1 = _darknet_conv(x,convs[0])
    for i in range(num_blocks):
        y = _darknet_conv(x1,convs[1])
        y = _darknet_conv(y,convs[2])
        x1 = add([x1, y])
    x1 = _darknet_conv(x1, convs[0])
    x = concatenate([x1,x2])
    x = _darknet_conv(x, convs[3])
    return x


def _residual_attention_block(input, convs, num_blocks, input_channels=None, output_channels=None, encoder_depth=1):
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1].value
    if output_channels is None:
        output_channels = input_channels

    if convs is None:
        raise ValueError("Assign a residual block function.")

    # First Residual Block

    # Trunk Branch
    output_trunk = input
    output_trunk = _csp_darknet_block(input=output_trunk, convs=convs, num_blocks=num_blocks)

    # Soft Mask Branch

    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  # 2倍降采样，特征图大小缩小半，512x512-->256x256
    for i in range(r):
        output_soft_mask = _csp_darknet_block(input=output_soft_mask, convs=convs, num_blocks=1)

    skip_connections = []
    for i in range(encoder_depth - 1):

        ## skip connections
        output_skip_connection = _csp_darknet_block(input=output_soft_mask, convs=convs, num_blocks=1)
        skip_connections.append(output_skip_connection)   # 0,1,2/0,1/0
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = _csp_darknet_block(input=output_soft_mask, convs=convs, num_blocks=1)

    ## decoder
    skip_connections = list(reversed(skip_connections))   # 反转保存的skip_connections
    for i in range(encoder_depth - 1):
        ## upsampling
        for _ in range(r):
            output_soft_mask = _csp_darknet_block(input=output_soft_mask,convs=convs, num_blocks=1)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        ## skip connections
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])

    ### last upsampling
    for i in range(r):
        output_soft_mask = _csp_darknet_block(input=output_soft_mask,convs=convs, num_blocks=1)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)

    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block

    return output