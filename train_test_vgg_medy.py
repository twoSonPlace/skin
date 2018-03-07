'''

Train and test a CNN with the VGG-16 pretrained feature extraction on a small images dataset.
Indepedent 20 runs are averaged and the resulting ROC and precision-and-recall curves are generated.

'''

import numpy as np
import ConfigParser 
import random
import string
import glob
import os
import ntpath

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.models import model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils

#scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.utils   import shuffle

# image processing
from scipy.ndimage.filters import gaussian_filter
from scipy import interp

# plot
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, './lib/')
from help_functions import *
from pre_processing_v2 import *


np.random.seed(15928)


#---
# input images
#---
img_width, img_height = 250/2, 200/2 # 

img_channels = 3   
nb_classes = 2    # classes: Grade3, Grade4 

# ---
# training conditions
# ---
# a ratio of test data to the total no. of data 
ratio_test_data = 0.15
ratio_validation_data = 0.25

num_runs = 20
nb_epoch = 40

data_augmentation = False
batch_size = 32

name_experiment = 'facial'
path_experiment = './' + name_experiment+ '/' 

INPUT_FILE_PATTERN = '*.bmp'        # original one


# ---
#--Path of the images 
# ---
dataset_path = './input/'

# positive and negative data samples
IMG_FOLDER = './datasets_for_train_test/'

img_train_file = name_experiment + '_dataset_imgs_train.hdf5'
label_train_file =  name_experiment + '_dataset_groundTruth_train.hdf5'
img_test_file = name_experiment + '_dataset_imgs_test.hdf5'
label_test_file =  name_experiment + '_dataset_groundTruth_test.hdf5'


#---------------------------------------------------------------------------------------------
def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def get_datasets_train_test(Nimgs, imgs_dir, ratio_for_test):

    groundTruth = np.zeros(Nimgs)
    tag_for_train = np.zeros(Nimgs)

    imgs = np.empty((Nimgs,img_height,img_width,img_channels))
    
    tmp = 0
    label_index = 0        # 'negative' directory appears first, so label 0 is set to the negative
    index_accumulated = 0
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path

        print '\nFolder: ', path
        if files:

            files = [ fi for fi in files if not fi.endswith(".png") ]        # exclude '*.png'

            for i in range(len(files)):
                #print files[i]
                full_filenm = os.path.join(path, files[i])    
                #print str(i) + ': ' + str(index_accumulated + i) + " : " + full_filenm
                img = Image.open(full_filenm)

                #
                # image pre-processing: cropping and resizing is made for finer ROI 
                #
                img = img.crop(((img.size[0]/6), (img.size[1]/3), img.size[0], img.size[1]))   # width comes first than height. 
                img = img.resize((img_width, img_height))

                #
                # when BW images are input images
                #
                img = np.asarray(img)
                if img.ndim == 2:
                     img2 = np.stack((img,)*3)    # forced to be converted into RGB color format.
                     img2 = np.transpose(img2, (1,2,0))       
                     img_chk = Image.fromarray(img2, 'RGB') # redundant, but to make it sure
                     img = np.asarray(img_chk)

                imgs[index_accumulated + i] = img
                groundTruth[index_accumulated + i] = label_index

                # splitting the data
                if np.random.rand() > ratio_for_test:
                    tag_for_train[index_accumulated + i] = 1
                else:
                    tag_for_train[index_accumulated + i] = 0

            tmp = len(files)
            label_index = label_index + 1
            print "   label : " + str(label_index - 1)

        index_accumulated = index_accumulated + tmp

    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))    # when handling grayscale images... that should be made for my_preprocessing..

    # image processing in which RGB color images are converted into grayscale ones.
    imgs = my_PreProc(imgs) 

    # -----
    # The part for making VGG-16 network work for grayscale images 
    # -----
    imgs = 255 * imgs  # rescaling is made because pixel values have been normalized

    # copy grayscale into three channels
    imgs_proc = np.zeros((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    for i in range(3):
        imgs_proc[:,i,:,:] = imgs[:,0,:,:]


    index = np.nonzero(tag_for_train == 1)[0]
    imgs_train = imgs_proc[index,:,:,:]
    groundTruth_train = groundTruth[index]

    index = np.nonzero(tag_for_train == 0)[0]
    imgs_test = imgs_proc[index,:,:,:]
    groundTruth_test = groundTruth[index]

    return imgs_train, groundTruth_train, imgs_test, groundTruth_test 


#--------------- run ------------------------------------------------------------------------- 

# variables for plot roc and precision curves
tprs = [[], []]
precisions = [[], []]
base_fpr = np.linspace(0,1,101)
base_recall = np.linspace(0,1,101)

#-------
# reading data 
#-------
nb_images = len(glob.glob(os.path.join(IMG_FOLDER, '*', INPUT_FILE_PATTERN)))
print('No. of image data: {}'.format(nb_images))

for run in range(num_runs):

    print '\n---------------------' + str(run) + '-th run ----------------------------\n'

    #-------
    # spliting the data into training and test. writing training and test images into hd5 files 
    #-------
    imgs_train, groundTruth_train, imgs_test, groundTruth_test = get_datasets_train_test(nb_images, IMG_FOLDER, ratio_test_data) 

    print('No. of image data for training: {}'.format(imgs_train.shape[0]))
    print('No. of image data for test: {}'.format(imgs_test.shape[0]))

    write_hdf5(imgs_train, dataset_path + img_train_file)
    write_hdf5(groundTruth_train, dataset_path + label_train_file)
    write_hdf5(imgs_test, dataset_path + img_test_file)
    write_hdf5(groundTruth_test, dataset_path + label_test_file)
    #print '\nrandom shuffling and splitting the data and then writing are done'

    #-------
    # training CNN with a training file 
    #-------

    #- CNN architecture -

    cnn_img_channels = 3    # For  VGG-16 network

    if  K.image_data_format() == 'channels_first':
        channel_axis = 1
        input = Input(shape=(cnn_img_channels, img_height, img_width), name='image_input')
    else:
        channel_axis = 3
        input = Input(shape=(img_height, img_width, cnn_img_channels), name='image_input')
    print 'image_dim_ordering = ' + str(channel_axis)

    # Get back the conv. part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(input_shape=(3, img_height, img_width), weights='imagenet', include_top=False, pooling=max)

    #--
    # freeze the first 10 layers of VGG16
    #--
    for layer in model_vgg16_conv.layers[:13]:
       layer.trainable = False 
    for layer in model_vgg16_conv.layers[13:]:
       layer.trainable = True 

    model_vgg16_conv.summary()

    # use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    # add the FC layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.20)(x)
    x = BatchNormalization()(x)
    x = Dense(nb_classes, activation='softmax', name='prediction')(x)

    # create my own model
    model = Model(input=input, output=x)
    model.summary()


    #sgd = SGD(lr=0.0015, decay=1e6, momentum=0.025, nesterov=True) # let's train the model using SGD + momentum (how original).
    adam = Adam(lr=0.001, beta_1=0.03, beta_2=0.499, epsilon=None, decay=0.00001, amsgrad=True)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


    X_train = load_hdf5(dataset_path + img_train_file)        # Normalized into 0 ~ 1 already.
    y_train = load_hdf5(dataset_path + label_train_file)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    X_train = X_train.astype('float32')
    Y_train = np_utils.to_categorical(y_train, nb_classes) # convert class vectors to binary class matrices

    json_string = model.to_json()
    open(path_experiment + name_experiment + '_' + str(nb_classes) + '_cnn_architecture.json', 'w').write(json_string)
    weight_filenm = name_experiment + '_' + str(nb_classes) + '_cnn_best_weight_' + str(run) + '.h5'

    checkpointer = ModelCheckpoint(filepath=path_experiment + weight_filenm, verbose=1, monitor='val_loss' , mode='auto', save_best_only=True) #save at each epoch if the validation decreased
    # 'val_acc' is much worse than 'val_loss'
    
    #data_augmentation = False
    #print data_augmentation
    if not data_augmentation:

        print('Not using data augmentation.')
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True, validation_split=ratio_validation_data, callbacks=[checkpointer])

    else:
        print('Using real-time data augmentation.')
        '''
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), samples_per_epoch=X_train.shape[0], epochs=nb_epoch, verbose=, shuffle=True, validation_data=(X_train, Y_train))
        '''        

    model.save_weights(path_experiment + name_experiment + '_' + str(nb_classes) + '_cnn_last_weight_' + str(run) + '.h5', overwrite=True)


    # --
    # test the network trained 
    # --
    X_test = load_hdf5(dataset_path + img_test_file)
    y_test = load_hdf5(dataset_path + label_test_file)

    print('X_test shape:', X_test.shape)
    print(X_test.shape[0], 'test samples')

    X_test = X_test.astype('float32')
    Y_test = np_utils.to_categorical(y_test, nb_classes) # convert class vectors to binary class matrices

    best_or_last = ['best', 'last']
    for idx in range(2):

        model.load_weights(path_experiment + name_experiment + '_' + str(nb_classes) + '_cnn_' + best_or_last[idx] + '_weight_' + str(run) + '.h5')
        y_pred = model.predict(X_test, batch_size=32, verbose=2)
        y_class = [row[1] for row in y_pred] 

        # roc
        plt.figure(3*idx)
        fpr, tpr, thresholds = roc_curve(y_test, y_class)
        plt.plot(fpr, tpr, 'b', alpha=0.2)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs[idx].append(tpr)

        # precision-recall
        plt.figure(3*idx+1)
        precision, recall, thresholds = precision_recall_curve(y_test, y_class)
        prec = np.fliplr([precision])[0]  # the array is increasing 
        recall = np.fliplr([recall])[0]  # the array is increasing 
        plt.plot(recall, prec, 'b', alpha=0.2)
        prec = interp(base_recall, recall, prec) 
        precisions[idx].append(prec)


#--
# End of training and test. Plotting the mean curves
#--
for idx in range(2):

   # roc curve
   tprs[idx] = np.array(tprs[idx])
   mean_tprs = tprs[idx].mean(axis=0)
   std = tprs[idx].std(axis=0)

   tprs_upper = np.minimum(mean_tprs + std, 1)
   tprs_lower = mean_tprs - std

   plt.figure(3*idx)
   plt.plot(base_fpr, mean_tprs, 'b')
   plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='gray', alpha=0.3)
   plt.plot([0,1], [0,1],'r--')
   plt.xlim([0, 1.00])
   plt.ylim([0, 1.00])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.axes().set_aspect('equal', 'datalim')
   plt.grid()
   #plt.show()
   plt.savefig(path_experiment+"ROC_" + best_or_last[idx] + '_wgt' + '.png')

   # precision-recall curve
   precisions[idx] = np.array(precisions[idx])
   mean_precs = precisions[idx].mean(axis=0)
   std = precisions[idx].std(axis=0)

   precs_upper = np.minimum(mean_precs + std, 1)
   precs_lower = mean_precs - std

   plt.figure(3*idx+1)
   plt.plot(base_recall, mean_precs, 'b')
   plt.fill_between(base_recall, precs_lower, precs_upper, color='gray', alpha=0.3)
   plt.xlim([0, 1.00])
   plt.ylim([0.5, 1.00])
   plt.ylabel('Precision')
   plt.xlabel('Recall')
   plt.axes().set_aspect('equal', 'datalim')
   plt.grid()
   #plt.show()
   plt.savefig(path_experiment+"PRECISION_" + best_or_last[idx] + '_wgt' + '.png')


