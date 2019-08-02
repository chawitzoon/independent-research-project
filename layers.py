#bilinearUpsampling2d layer heavily based on DenseDepth papar
import tensorflow as tf
import keras.utils.conv_utils as conv_utils
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K

#bilinearUpstream2D is not trainable, no weight to train, no need for build()
class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
        width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        height = self.size[0] * input_shape[1] if inputs.shape[1] is not None else None
        width = self.size[1] * input_shape[2] if inputs.shape[2] is not None else None
        
        #https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image/resize
        return tf.image.resize(inputs, size=[height,width], method=tf.image.ResizeMethod.BILINEAR,antialias=True,)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))