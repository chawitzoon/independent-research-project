import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

#reference from eigen.etal
def scale_invariant_function(y_true, y_pred):
    # def scale_invariant(y_true,y_pred):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)

    y_true:  one depth map
    y_pred:  another depth map

    Returns: 
        scale_invariant_distance

    """
    # sqrt(Eq. 3)
#         assert(np.all(tf.math.is_finite(depth1) & tf.math.is_finite(depth2) & (depth1 > 0) & (depth2 > 0)))
    log_diff = y_pred - K.log(y_true)
    num_pixels = float(log_diff.get_shape()[1]*log_diff.get_shape()[2])

    if num_pixels == 0:
        return np.nan
    else:
        return K.sum(K.sum(K.square(log_diff), axis=(1, 2, 3), keepdims=False) / num_pixels  - 0.5 * K.square(K.sum(log_diff, axis=(1, 2, 3), keepdims=True)) / K.square(num_pixels))
    # return scale_invariant

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=10.0):
    # def depth_loss(y_true, y_pred):

    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))
    
    # return depth_loss

def semantic_loss_function(y_true, y_pred):
    # def semantic_loss(y_true, y_pred):

    semantic_loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return semantic_loss_value

    # return semantic_loss

# def joint_loss_function(y_true1, y_pred1, y_true2, y_pred2, loss_weight=0.7):
#     loss = loss_weight*depth_loss_function(y_true1, y_pred1) + (1 - loss_weight)*semantic_loss_function(y_true2, y_pred2)
#     return loss