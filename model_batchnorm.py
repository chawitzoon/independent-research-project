import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Softmax, BatchNormalization
from tensorflow.keras import layers
from IPython.display import Image

from layers import BilinearUpSampling2D

def create_model(transfer = 'finetune', semantic_joint = True):
    print('Loading base model (DenseNet)..')
    if transfer == 'random':
        base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False, weights=None)
    else:
        base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False, weights='imagenet')
    # base_model.summary()
    # tf.keras.utils.plot_model(base_model, 'base_model_with_shape_info.png', show_shapes=True)
    # Image(retina=True, filename='base_model_with_shape_info.png')
    
    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    for layer in base_model.layers:
        if transfer == 'finetune' or transfer == 'random':
            layer.trainable = True
        else:
            layer.trainable = False
    

    # Starting number of decoder filters, make half filter (half channel)
    decode_filters = int(int(base_model_output_shape[-1])/2)

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
        up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
        up_i = BatchNormalization()(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
        up_i = BatchNormalization()(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.layers[-1].output)

    decoder = upproject(decoder, int(decode_filters/2),'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')

    if semantic_joint == True:
        # Extract depths (final layer)
        conv3_depth = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3_depth')(decoder)
        # Extract semantic (final layer)
        conv3_semantic = Conv2D(filters=151, kernel_size=3, strides=1, padding='same', name='conv3_semantic')(decoder)
        conv3_semantic = Softmax(axis = -1,name = 'softmax_semantic')(conv3_semantic)
        # Create the model
        model = Model(inputs=base_model.input, outputs= [conv3_depth,conv3_semantic])
        model.summary()
        # tf.keras.utils.plot_model(model, 'model_with_shape_info.png', show_shapes=True)
        # Image(retina=True, filename='model_with_shape_info.png')
        
    else:
        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)
        # Create the model
        model = Model(inputs=base_model.input, outputs=conv3)
        model.summary()
        # tf.keras.utils.plot_model(model, 'model_with_shape_info.png', show_shapes=True)
        # Image(retina=True, filename='model_with_shape_info.png')
    return model