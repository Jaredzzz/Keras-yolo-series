from keras.layers import Input
from keras.engine.topology import Layer
from keras import backend as K


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
        tanh=(1 - e^{-2x})/(1 + e^{-2x})
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
         X_input = Input(input_shape)
         X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        
        return input_shape

class Mish6(Layer):
    '''
    Mish6 Activation Function.
    '''

    def __init__(self, **kwargs):
        super(Mish6, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        x1 = tf.minimum(tf.maximum(0.0, inputs * K.tanh(K.softplus(inputs))), 6.0)
        x2 = inputs * K.tanh(K.softplus(inputs))
        x = tf.where(inputs > 0.0, x1, x2)
        return x

    def get_config(self):
        config = super(Mish6, self).get_config()
        return config

    def compute_output_shape(self, input_shape):

        return input_shape

