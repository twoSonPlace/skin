'''

Train and test a simple deep CNN on the small images dataset.
The image dataset is divided for training and test. 
Nearly 10-fold validation. 10 runs are made and their results are averaged.

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
from keras.layers import Input, Dense, Dropout, Activation, Flatten
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

# plot
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, './lib/')
from help_functions import *
from pre_processing_v2 import *


np.random.seed(15928)

best_or_last = 'last'

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
nb_epoch = 8000

data_augmentation = False
batch_size = 32

name_experiment = 'facial'
path_experiment = './' + name_experiment+ '/' 

INPUT_FILE_PATTERN = '*.bmp'        # original one
#INPUT_FILE_PATTERN = '*.png'        # Frangi filered 


#------------Path of the images --------------------------------------------------------------

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

    imgs = np.empty((Nimgs,img_height,img_width,img_channels))
    groundTruth = np.zeros(Nimgs)
    tag_for_train = np.zeros(Nimgs)
    
    tmp = 0
    label_index = 0        # 'negative' directory appears first, so label 0 is set to the negative
    index_accumulated = 0
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path

        print '\nFolder: ', path
        if files:

            #files = [ fi for fi in files if not fi.endswith(".bmp") ]        # exclude '*.bmp'
            files = [ fi for fi in files if not fi.endswith(".png") ]        # exclude '*.png'

            for i in range(len(files)):
                #print files[i]
                full_filenm = os.path.join(path, files[i])    
                #print str(i) + ': ' + str(index_accumulated + i) + " : " + full_filenm
                img = Image.open(full_filenm)
                img = img.resize((img_width, img_height))

                # check if image read does not fail
                #img.show()

                img = np.asarray(img)
                if img.ndim == 2:
                     img = np.stack((img,)*3)    # forced to be converted into RGB color format.
                     #img = np.transpose(img, (0,1,2))
                     img = np.transpose(img, (1,2,0))       # when handling grayscale images

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
    print imgs.shape

    # to examine the convered images
    '''
    num_imgs = imgs.shape[0]
    for i in range(num_imgs):
        array = 255*imgs[i,0,:,:]    # In a my_PreProc function, values between 0 and 1 are return. To visualize .. 
        img2 = Image.fromarray(array.astype(np.uint8))
        img2.save( str(i) + '_chk.gif')
    exit()
    '''

    # random shuffle
    imgs, groundTruth, tag_for_train = shuffle(imgs, groundTruth, tag_for_train, random_state=0)

    index = np.nonzero(tag_for_train == 1)[0]
    imgs_train = imgs[index,:,:,:]
    groundTruth_train = groundTruth[index]

    index = np.nonzero(tag_for_train == 0)[0]
    imgs_test = imgs[index,:,:,:]
    groundTruth_test = groundTruth[index]

    return imgs_train, groundTruth_train, imgs_test, groundTruth_test 


#--------------- run ------------------------------------------------------------------------- 

#-------
# reading data 
#-------
nb_images = len(glob.glob(os.path.join(IMG_FOLDER, '*', INPUT_FILE_PATTERN)))
print('No. of image data: {}'.format(nb_images))

for run in range(num_runs):

    print '\n---------------------' + str(run) + '-th run ------------------------------------------\n'

    #-------
    # spliting the data into training and teste
    # writing training and test images into hd5 files 
    #-------
    imgs_train, groundTruth_train, imgs_test, groundTruth_test = get_datasets_train_test(nb_images, IMG_FOLDER, ratio_test_data) 

    print('No. of image data for training: {}'.format(imgs_train.shape[0]))
    print('No. of image data for test: {}'.format(imgs_test.shape[0]))

    #write_hdf5(imgs_train, dataset_path + img_train_file)
    #write_hdf5(groundTruth_train, dataset_path + label_train_file)
    write_hdf5(imgs_test, dataset_path + img_test_file)
    write_hdf5(groundTruth_test, dataset_path + label_test_file)
    #print '\nrandom shuffling and splitting the data and then writing are done'


    # test the network trained 
    X_test = load_hdf5(dataset_path + img_test_file)
    y_test = load_hdf5(dataset_path + label_test_file)

    print('X_test shape:', X_test.shape)
    print(X_test.shape[0], 'test samples')

    X_test = X_test.astype('float32')
    Y_test = np_utils.to_categorical(y_test, nb_classes) # convert class vectors to binary class matrices

    # load the best model trained 
    model = model_from_json(open(path_experiment + name_experiment + '_' +str(nb_classes) + '_cnn_architecture.json').read())
    model.load_weights(path_experiment + name_experiment + '_' + str(nb_classes) + '_cnn_' + best_or_last + '_weight_' + str(run) + '.h5')


    y_pred = model.predict(X_test, batch_size=32, verbose=2)
    #y_class = [ np.argmax(row) for row in y_pred] 
    #print zip(y_pred, y_class)
    y_class = [ row[1] for row in y_pred] 
    #print zip(y_pred, y_class)

    # file output
    fp = open(path_experiment +'cnn_output_2_' + str(run) + '.txt', 'w')
    output = zip(y_class, y_test)
    for row in output:
        for field in row:
            fp.write('{} '.format(field))
        fp.write('\n')
    fp.close()


    '''
    y_scores = [ row[1] for row in y_pred ]
    
    # ROC curve of two-classes classification problem
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    AUC_ROC = roc_auc_score(y_test, y_scores)
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC_Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')
    plt.savefig(path_experiment+"ROC_" + str(run) + ".png")

    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision,recall)
    prec_rec_curve = plt.figure()
    plt.plot(recall,precision,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(path_experiment+"Precision_recall_" + str(run) + ".png")

    plt.close("all")
    '''
