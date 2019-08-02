import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import os
import h5py
from skimage.measure import block_reduce
from skimage.transform import resize


# image_path = os.path.join(os.getcwd(),'all/npy_combined_image/npy_image.npy'
def load_images(image_path):
    images_nparray = np.load(image_path)
    return images_nparray

# depth_path = os.path.join(os.getcwd(),'all/npy_combined_depth/npy_depth.npy')
def load_depths(depth_path):
    depths_nparray = np.load(depth_path)
    return depths_nparray

# label_path = os.path.join(os.getcwd(),'all/npy_combined_depth/npy_label.npy')
def load_labels(label_path):
    labels_nparray = np.load(label_path)
    return labels_nparray

# test_data_path = os.path.join(os.getcwd(),'nyu_depth_v2_labeled.mat')
def load_test_data(test_data_path):
    # read from .mat file
    with h5py.File(test_data_path,'r') as f:
        depths = f[('depths')]
        images = f[('images')]

        # f is h5py file
        # read entire h5py Dataset to numpy array
        images_nparray_unorder = f['images'][()]
        depths_nparray_unorder = f['depths'][()]
        
        #reorder (N,C,W,H) > (N,W,H,C)
        images_nparray_unorder1 = np.rollaxis(images_nparray_unorder, 1, 4)
        # #reorder (N,W,H,C) > (N,H,W,C)
        images_nparray = np.rollaxis(images_nparray_unorder1, 1, 3)
        depths_nparray = np.expand_dims(np.rollaxis(depths_nparray_unorder, 1, 3), axis = -1)
        depths_nparray = block_reduce(depths_nparray, block_size=(1, 2, 2, 1), func=np.max)
        print(images_nparray.shape)
        print(depths_nparray.shape)
        return images_nparray, depths_nparray

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#create imagedatagen for train, validation and test dataset
def create_data_generator(images_nparray, depths_nparray, labels_nparray, batch_size = 5, training = True, semantic_joint = True):
    
    if semantic_joint == True and training == True:
        def _combine_generator(gen1, gen2, gen3):
            while True:
                yield(gen1.next(), [gen2.next(), gen3.next()])
    else:
        def _combine_generator(gen1, gen2):
            while True:
                yield(gen1.next(), gen2.next())
    
    #create ImagDataGenerator
    seed = 1
    if training == True:
        image_datagen = ImageDataGenerator(
                        rescale = 1/255,
                        horizontal_flip = True,
                        validation_split=0.18)
        depth_datagen = ImageDataGenerator(
                        horizontal_flip = True,
                        validation_split=0.18)
        if semantic_joint == True:
            label_datagen = ImageDataGenerator(
                        horizontal_flip = True,
                        validation_split=0.18)
    else:
        image_datagen = ImageDataGenerator(
                        rescale = 1/255,
                        horizontal_flip = False)
        depth_datagen = ImageDataGenerator(
                        horizontal_flip = False)
              
    print('ImageDataGenerator created')
        
    # https://keras.io/preprocessing/image/
    # image_datagen.fit(images_nparray, augment=True, seed=seed)
    # depth_datagen.fit(depths_nparray, augment=True, seed=seed)
    # if semantic_joint == True and training == True:
    #     label_datagen.fit(labels_nparray, augment=True, seed=seed)

    # print('datagen fitted')
    
    # flow
    if training == True:
        image_generator_t = image_datagen.flow(
        images_nparray, batch_size=batch_size, subset='training',
        seed=seed)

        depth_generator_t = depth_datagen.flow(
        depths_nparray, batch_size=batch_size, subset='training',
        seed=seed)

        if semantic_joint == True:
            label_generator_t = label_datagen.flow(
                labels_nparray, batch_size=batch_size, subset='training',
                seed=seed)

        image_generator_v = image_datagen.flow(
        images_nparray, batch_size=batch_size, subset='validation',
        seed=seed)

        depth_generator_v = depth_datagen.flow(
        depths_nparray, batch_size=batch_size, subset='validation',
        seed=seed)

        if semantic_joint == True:
            label_generator_v = label_datagen.flow(
                labels_nparray, batch_size=batch_size, subset='validation',
                seed=seed)
    else:
        image_generator_test = image_datagen.flow(
        images_nparray, batch_size=batch_size, shuffle=False, seed=seed)
        
        depth_generator_test = depth_datagen.flow(
        depths_nparray, batch_size=batch_size, shuffle=False, seed=seed)

    print('generator flowed')

    #generator
    if training == True:
        if semantic_joint == True:
            train_generator = _combine_generator(image_generator_t, depth_generator_t, label_generator_t)
            validation_generator = _combine_generator(image_generator_v, depth_generator_v, label_generator_v)
        else:
            train_generator = _combine_generator(image_generator_t, depth_generator_t)
            validation_generator = _combine_generator(image_generator_v, depth_generator_v)
        return train_generator, validation_generator
    else:
        test_generator = _combine_generator(image_generator_test, depth_generator_test)
        return test_generator

    print('finished creating datagenerator')

# def create_data_generator_eigen(images_nparray, depths_nparray, labels_nparray, batch_size = 5, training = True, semantic_joint = False):

#     images_nparray = block_reduce(images_nparray, block_size=(1, 2, 2, 1), func=np.max)
#     depths_nparray = block_reduce(depths_nparray, block_size=(1, 2, 2, 1), func=np.max)

#     def _random_crop_resize(img, depth):
    
#         assert img.shape[2] == 3
#         a = np.random.randint(12)
#         b = np.random.randint(16)

#         img = img[a:a-12,b:b-16,:]
#         depth = depth[a:a-12,b:b-16,:]
#         depth_down = resize(depth,(55, 74, 1))
        
#         #no need to divide by 10 because using log
#         return img, depth_down

#     def _crop_generator(batch_x, batch_y):
#         """Take as input a Keras ImageGen (Iterator) and generate random
#         crops from the image batches generated by the original iterator.
#         """
#         while True:
#             batch_x= next(batch_x)
#             batch_y = next(batch_y)
#             batch_crops = np.zeros((batch_x.shape[0], 228, 304, 3))
#             batch_crops_y = np.zeros((batch_y.shape[0], 55, 74, 1))
#             for i in range(batch_x.shape[0]):
#                 batch_crops[i], batch_crops_y[i] = _random_crop_resize(batch_x[i], batch_y[i])
#             yield batch_crops, batch_crops_y

#     #create ImagDataGenerator
#     seed = 1
#     if training == True:
#         image_datagen = ImageDataGenerator(
#                         rescale = 1/255,
#                         horizontal_flip = True,
#                         validation_split=0.18)
#         depth_datagen = ImageDataGenerator(
#                         horizontal_flip = True,
#                         validation_split=0.18)
#     else:
#         image_datagen = ImageDataGenerator(
#                         rescale = 1/255,
#                         horizontal_flip = False)
#         depth_datagen = ImageDataGenerator(
#                         horizontal_flip = False)
                
#     print('ImageDataGenerator created')
        
#     # # https://keras.io/preprocessing/image/
#     # image_datagen.fit(images_nparray, augment=True, seed=seed)
#     # depth_datagen.fit(depths_nparray, augment=True, seed=seed)
#     # if semantic_joint == True and training == True:
#     #     label_datagen.fit(labels_nparray, augment=True, seed=seed)

#     # print('datagen fitted')
    
#     # flow
#     if training == True:
#         image_generator_t = image_datagen.flow(
#         images_nparray, batch_size=batch_size, subset='training',
#         seed=seed)

#         depth_generator_t = image_datagen.flow(
#         depths_nparray, batch_size=batch_size, subset='training',
#         seed=seed)

#         image_generator_v = image_datagen.flow(
#         images_nparray, batch_size=batch_size, subset='validation',
#         seed=seed)

#         depth_generator_v = image_datagen.flow(
#         depths_nparray, batch_size=batch_size, subset='validation',
#         seed=seed)
#     else:
#         generator_test = image_datagen.flow(
#         images_nparray, batch_size=batch_size, shuffle=False, seed=seed)

#         generator_test = image_datagen.flow(
#         depths_nparray, batch_size=batch_size, shuffle=False, seed=seed)
#     print('generator flowed')

#     if training == True:
#         image_generator_t_crop, depth_generator_t_crop = _crop_generator(image_generator_t, depth_generator_t)
#         image_generator_v_crop, depth_generator_v_crop = _crop_generator(image_generator_v, depth_generator_v)
#     else:
#         image_generator_test_crop, depth_generator_test_crop = _crop_generator(generator_test)

#     if training == True:
#         while True:
#             train_generator = [image_generator_t_crop, image_generator_t_crop], depth_generator_t_crop
#             validation_generator = [image_generator_v_crop, image_generator_v_crop], depth_generator_v_crop
#         return train_generator, validation_generator
#     else:
#         while True:
#             test_generator = [image_generator_test_crop, image_generator_test_crop], image_generator_test_crop
#         return test_generator

    # print('finished creating datagenerator')




# Callbacks for loss in every setp per epoch
# class LossHistory(keras.callbacks.Callback):
#     def __init__(self, semantic_joint):
#         self.semantic_joint = semantic_joint

#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.conv3_depth_losses = []
#         self.softmax_semantic_losses = []
#         self.val_losses = []
#         self.val_conv3_depth_losses = []
#         self.val_softmax_semantic_losses = []
        
#     def on_batch_end(self, batch, logs={}):
#         if self.semantic_joint == True:
#             self.losses.append(logs.get('loss'))
#             self.conv3_depth_losses.append(logs.get('conv3_depth_loss'))
#             self.softmax_semantic_losses.append(logs.get('softmax_semantic_loss'))
#         else:
#             self.losses.append(logs.get('loss'))

#load training visualization
def save_train_validation_graph(history, name, semantic_joint, save_path):
    if semantic_joint == True:
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['conv3_depth_loss'])
        ax1.plot(history.history['softmax_semantic_loss'])
        ax1.plot(history.history['val_loss'])
        ax1.plot(history.history['val_conv3_depth_loss'])
        ax1.plot(history.history['val_softmax_semantic_loss'])
        # ax1.set_title('model loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'depth', 'semantic', 'val', 'val_depth', 'val_semantic'], loc='upper left')
        fig1.savefig(save_path + f'/history_{name}.png')

        # fig2, (ax2, ax3, ax4) = plt.subplots(nrows=3, ncols=1) # two axes on figure
        # ax2.plot(history_batch.losses)
        # ax2.set_ylabel('loss')
        # ax2.set_xlabel('step')

        # ax3.plot(history_batch.conv3_depth_losses)
        # ax3.set_ylabel('conv3_depth_loss')
        # ax3.set_xlabel('step')

        # ax4.plot(history_batch.softmax_semantic_losses)
        # ax4.set_ylabel('softmax_semantic_loss')
        # ax4.set_xlabel('step')
        # fig2.savefig(save_path + f'/history_step_{name}.png')

        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['conv3_depth_loss'])
        ax2.plot(history.history['val_conv3_depth_loss'])
        # ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['depth', 'val_depth'], loc='upper left')
        fig2.savefig(save_path + f'/history_depth_{name}.png')

    else:
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])
        ax1.set_title('model loss')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax1.legend(['depth', 'val_depth'], loc='upper left')
        fig1.savefig(save_path + f'/history_{name}.png')

        # fig2, ax2 = plt.subplots()
        # ax2.plot(history_batch.losses)
        # ax2.set_ylabel('loss')
        # ax2.set_xlabel('step')
        # fig2.savefig(save_path + f'/history_step_{name}.png')
    

#visualization
# def visualize(images_nparray, depths_nparray,labels_nparray, depths_pred, )
#visualize
def test_visualize(semantic_joint, save_path, number, name, images_nparray, depths_nparray, depths_pred, labels_nparray=None, labels_pred=None):
    if semantic_joint == True:
        for i in range(number):
            plt.imshow(images_nparray[i,:,:,:], interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/image{i}.png', )
            plt.clf()

            depths_nparray_visual = np.squeeze(depths_nparray)
            plt.imshow(depths_nparray_visual[i,:,:], interpolation='nearest',)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/groundtruth{i}.png', )
            plt.clf()

            depths_pred_visual = np.squeeze(depths_pred)
            plt.imshow(depths_pred_visual[i,:,:], interpolation='nearest', )
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/depth{i}_{name}.png', )
            plt.clf()

            depth_flatten = depths_nparray_visual[i,:,:].flatten()
            plt.hist(depth_flatten, bins = 100)
            plt.tight_layout()
            plt.gcf().savefig(save_path + f'/depth{i}_{name}_hist.png')
            plt.clf()

            labels_pred_visual = np.squeeze(np.argmax(labels_pred, axis = -1))
            plt.imshow(labels_pred_visual[i,:,:],interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/label{i}_{name}.png', )
            plt.clf()
    else:
        for i in range(number):
            plt.imshow(images_nparray[i,:,:,:], interpolation='nearest')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/image{i}.png', )
            plt.clf()

            depths_nparray_visual = np.squeeze(depths_nparray)
            plt.imshow(depths_nparray_visual[i,:,:], interpolation='nearest',)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/groundtruth{i}.png', )
            plt.clf()

            depth_flatten = depths_nparray_visual[i,:,:].flatten()
            plt.hist(depth_flatten, bins = 100)
            plt.tight_layout()
            plt.gcf().savefig(save_path + f'/groundtruth{i}_hist.png')
            plt.clf()

            depths_nparray_visual = np.squeeze(depths_pred)
            plt.imshow(depths_nparray_visual[i,:,:], interpolation='nearest',)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_path + f'/depth{i}_{name}.png', )
            plt.clf()

            depth_flatten = depths_nparray_visual[i,:,:].flatten()
            plt.hist(depth_flatten, bins = 100)
            plt.tight_layout()
            plt.gcf().savefig(save_path + f'/depth{i}_{name}_hist.png')
            plt.clf()