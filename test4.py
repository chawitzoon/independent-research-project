#example for train1
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from evaluate import evaluate
from utils import create_data_generator, load_test_data, test_visualize
from layers import BilinearUpSampling2D
from loss import depth_loss_function, semantic_loss_function
os.environ["CUDA_VISIBLE_DEVICES"]="4"


name = 'train4_batch'
semantic_joint = True
root_dir = os.getcwd()
test_data_path = os.path.join(root_dir,'all/nyu_depth_v2_labeled.mat')
save_path = './figures/train4_batch'
number = 20
batch_size = 5
# Custom object needed when loading model
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function, 'semantic_loss_function':semantic_loss_function}

#load model (include weight)
model = load_model(f'model_{name}.h5', custom_objects=custom_objects)
print('model loaded')

#load weight
# weight_path = './weights2/weights(1tran).12-0.39.hdf5'
# model.load_weights(weight_path)

#load data
images_nparray, depths_nparray = load_test_data(test_data_path)
images_nparray = images_nparray/255
print('data loaded')

#use model for prediction
# test_generator = create_data_generator(images_nparray, depths_nparray, None, training = False, semantic_joint = semantic_joint)
# if semantic_joint == True:
#     depths_pred, labels_pred = model.predict_generator(test_generator,steps = int(len(images_nparray)/batch_size))
# else:
#     depths_pred = model.predict_generator(test_generator,steps = int(len(images_nparray)/batch_size)) 
# print('prediction finished')

#use model alternative without datagen
if semantic_joint == True:
    depths_pred, labels_pred = model.predict(images_nparray, batch_size=batch_size)
else:
    depths_pred = model.predict(images_nparray, batch_size=batch_size)
print('prediction finished')

#evaluate the scores
error_scores = evaluate(model, images_nparray, depths_nparray, depths_pred, batch_size=6)
print(f'abs_rel:{error_scores[0]}')
print(f'rmse:{error_scores[1]}')
print(f'logrmse:{error_scores[2]}')
print(f'a1:{error_scores[3]}')
print(f'a2:{error_scores[4]}')
print(f'a3:{error_scores[5]}')


test_visualize(semantic_joint, save_path, number, name, images_nparray, depths_nparray, depths_pred, labels_pred=labels_pred)

#visualize
# for i in range(number):
#     plt.imshow(images_nparray[i,:,:,:], interpolation='nearest')
#     plt.axis('off')
#     plt.savefig(save_path + f'/image{i}.png', bbox_inches=0)

#     depths_nparray_visual = np.squeeze(depths_nparray)
#     plt.imshow(depths_nparray_visual[i,:,:], interpolation='nearest', vmin=0, vmax=12,)
#     plt.axis('off')
#     plt.savefig(save_path + f'/groundtruth{i}.png', bbox_inches=0)

#     depths_nparray_visual = np.squeeze(depths_pred)
#     plt.imshow(depths_nparray_visual[i,:,:], interpolation='nearest', vmin=0, vmax=12,)
#     plt.axis('off')
#     plt.savefig(save_path + f'/depth{i}_{name}.png', bbox_inches=0)

#     depth_flatten = depths_nparray_visual.flatten()
#     plt.hist(depth_flatten, bins = 100)
#     plt.savefig(save_path + f'/depth{i}_{name}_hist.png')

#     labels_pred_visual = np.squeeze(np.argmax(labels_pred, axis = -1))
#     plt.imshow(labels_pred_visual[i,:,:],interpolation='nearest')
#     plt.axis('off')
#     plt.savefig(save_path + f'/label{i}_{name}.png', bbox_inches=0)


