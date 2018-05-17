# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Activation, BatchNormalization, Conv2D, Input
from keras.layers.core import Dropout
from keras.layers.merge import Add
from keras.models import Model
from keras.utils import conv_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def res_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    """实例化Keras Resnet块。
    
    Arguments:
        input_tensor {[type]} -- 输入张量
        filters {[type]} -- filters
    
    Keyword Arguments:
        kernel_size {tuple} -- [description] (default: {(3,3)})
        strides {tuple} -- [description] (default: {(1,1)})
        use_dropout {bool} -- [description] (default: {False})
    """
    x = ReflectionPadding2D((1, 1))(input_tensor)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if use_dropout:
        x=Dropout(0.5)(x)
    
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)

    merged = Add()([input_tensor, x])
    return merged






def spatial_reflection_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    """填充4D张量的第二维和第三维.
    
    Arguments:
        x {[type]} -- [description]
    
    Keyword Arguments:
        padding {tuple} -- [description] (default: {((1,1),(1,1))})
        data_format {[type]} -- [description] (default: {None})
    """
    assert len(padding) == 2
    assert len(padding[0]) == 2
    assert len(padding[1]) == 2

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if data_format == "channels_first":
        pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
    else:
        pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]

    return tf.pad(x, pattern, "REFLECT")


class ReflectionPadding2D(Layer):
    """继承Layer
    
    Arguments:
        Layer {[type]} -- [description]
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]
    """


    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding,"__len__"):
            if len(padding) != 2:
                 raise ValueError('`padding` should have two elements. '
                                  'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2, "1st entry of padding")
            width_padding = conv_utils.normalize_tuple(padding[1], 2, "2nd entry of padding")
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))
            self.input_spec = InputSpec(ndim=4)
        
    def compute_output_shape(self,input_shape):
        """计算输出shape
        
        Arguments:
            input_shape {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """

        if self.data_format == "channels_first":
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == "channels_last":
            if input_shape[1] is not None:
                rows = input_shape[1]+self.padding[0][0]+self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2]+self.padding[1][0]+self.padding[1][1]
            else:
                cols = None
            return (input_shape[0], rows, cols, input_shape[3])
    
    def call(self, inputs):
        return spatial_reflection_2d_padding(inputs,
                                             padding=self.padding,
                                             data_format=self.data_format)

    def get_config(self):
        """配置
        
        Returns:
            [type] -- [description]
        """

        config = {"padding": self.padding, "data_format": self.data_format}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    input = Input(shape=(256, 256, 3))
    x = ReflectionPadding2D(3)(input)
    model = Model(input, x)
    model.summary()
