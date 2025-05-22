 # Functions for loading/preprocessing data

import os
import cv2
import numpy as np
from sklearn import preprocessing
import tensorflow.keras.layers as tfl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

def preprocess_images(dataset_path):
    images_data = []
    images_label = []
    class_names = os.listdir(dataset_path)
    for class_name in class_names:
        images_path = dataset_path + '/' + class_name
        images = os.listdir(images_path)
        for image in images:
            bgr_img = cv2.imread(images_path + '/' + image)
            # dsize
            dsize = (64,64)
            #resize image
            resized_image = cv2.resize(bgr_img,dsize)
            # convert from BGR color-space to YCrCb
            ycrcb_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YCrCb)
            # create a CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            # Now apply CLAHE object on the YCrCb image
            ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
            # convert back to BGR color-space from YCrCb
            equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
            # Denoise is done to remove unwanted noise to better perform
            equalized_denoised_image = cv2.fastNlMeansDenoisingColored(equalized_img, None, 10, 10, 7, 21)

            images_data.append(equalized_denoised_image/255)
            images_label.append(class_name)
    images_data = np.array(images_data)
    images_label = np.array(images_label)
    return images_data, images_label

def encoding_targets(labels):
    le = preprocessing.LabelEncoder()
    images_label = le.fit_transform(labels)
    return images_label

def dataAugmentation():
    dataAugmentation = ImageDataGenerator(rotation_range = 10, zoom_range = 0.30,
                                            fill_mode = "nearest", shear_range = 0.30)
    return dataAugmentation

def training_model_2(train_images_data, train_images_label, model, kfold, dataAugmentation):
    # Define per-fold score containers
    val_acc_per_fold = []
    val_loss_per_fold = []
    loss_per_fold = []
    acc_per_fold = []

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, valid in kfold.split(train_images_data, train_images_label):
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        history = model.fit(
            dataAugmentation.flow(train_images_data[train], train_images_label[train], batch_size=16),
            epochs=20,
            validation_data=(train_images_data[valid], train_images_label[valid])
        )
        val_acc_per_fold.append(history.history['val_accuracy'])
        acc_per_fold.append(history.history['accuracy'])
        val_loss_per_fold.append(history.history['val_loss'])
        loss_per_fold.append(history.history['loss'])
        # Increase fold number
        fold_no += 1
    return val_acc_per_fold,val_loss_per_fold,loss_per_fold,acc_per_fold


