import tensorflow as tf
from model_batchnorm import create_model
from loss import depth_loss_function, semantic_loss_function
from utils import *
from matplotlib import pyplot as plt
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

name = 'train4_batch'
transfer = 'finetune'
semantic_joint = True
root_dir = os.getcwd()
image_path = os.path.join(root_dir,'all/npy_combined_image/npy_image.npy')
depth_path = os.path.join(root_dir,'all/npy_combined_depth/npy_depth.npy')
label_path = os.path.join(root_dir,'all/npy_combined_label/npy_label.npy')
weight_path = None
save_path = './figures/train4_batch'

# Create Callbacks
lr_reduce = ReduceLROnPlateau(monitor='loss',factor=0.01,patient=5)

# Create model
model = create_model(transfer = transfer, semantic_joint = semantic_joint)

# Compile
if semantic_joint == True:
    model.compile(loss={'conv3_depth': depth_loss_function,
                            'softmax_semantic': semantic_loss_function},
                    loss_weights={'conv3_depth': 1,
                            'softmax_semantic': 0.2},
                    optimizer='Adagrad')
else:
    model.compile(loss=depth_loss_function,
                optimizer='Adagrad')


#load data
images_nparray = load_images(image_path)
depths_nparray = load_depths(depth_path)
if semantic_joint == True:
    labels_nparray = load_labels(label_path)
    train_generator, validation_generator = create_data_generator(images_nparray, depths_nparray, labels_nparray, training = True, batch_size = 5, semantic_joint = semantic_joint)
else:
    train_generator, validation_generator = create_data_generator(images_nparray, depths_nparray, None, training = True, batch_size = 5, semantic_joint = semantic_joint)

history = model.fit_generator(
train_generator,
steps_per_epoch = 2061, #train_generator.samples // batch_size
validation_data = validation_generator,
validation_steps = 452, #validation_generator.samples // batch_size
epochs = 50,
callbacks=[lr_reduce],)

model.save(f'model_{name}.h5')
save_train_validation_graph(history, name, semantic_joint, save_path)

with open(save_path + '/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)