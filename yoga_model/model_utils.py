 # Functions for building, training, saving model

import tensorflow.keras.layers as tfl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import data_utils as d
import warnings
import os
warnings.filterwarnings("ignore")

#unzip the dataset file 
#!kaggle datasets download -d niharika41298/yoga-poses-dataset

path ="C:/Users/hadhr/Documents/yoga-project/data"
train_path = path + "/DATASET/TRAIN"
test_path = path + "/DATASET/TEST"

train_images_data, train_images_label = d.preprocess_images(train_path)

class_names = os.listdir(train_path)
class_num = len(class_names)
train_images_label = d.encoding_targets(train_images_label)
print("classes" , class_names)



model = tf.keras.Sequential([
        tfl.Conv2D(filters=16, kernel_size=(3,3), activation='relu',input_shape=(64,64,3)),
        tfl.MaxPool2D(pool_size=(2,2)),
        tfl.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tfl.BatchNormalization(axis=-1),
        tfl.Dropout(rate=0.25),

        tfl.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tfl.MaxPool2D(pool_size=(2,2)),
        tfl.BatchNormalization(axis=-1),
        tfl.Dropout(rate=0.25),

        tfl.Flatten(),
        tfl.Dense(512,activation='relu'),
        tfl.BatchNormalization(),
        tfl.Dropout(rate=0.5),
        tfl.Dense(class_num, activation='softmax')
])

model.compile(
	optimizer = 'adam',
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy']
)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True,random_state=2)

val_acc_per_fold,val_loss_per_fold,loss_per_fold,acc_per_fold=d.training_model(train_images_data, train_images_label, model, kfold, dataAugmentation)


print(f'> Mean_Training_Accuracy: {np.mean(acc_per_fold)*100} (+- {np.std(acc_per_fold)})')
print(f'> Mean_Validation_Accuracy: {np.mean(val_acc_per_fold)*100} (+- {np.std(val_acc_per_fold)})')




val_acc_per_fold,val_loss_per_fold,loss_per_fold,acc_per_fold=d.training_model(train_images_data, train_images_label, model, kfold, dataAugmentation)


print(f'> Mean_Training_Accuracy: {np.mean(acc_per_fold)*100} (+- {np.std(acc_per_fold)})')
print(f'> Mean_Validation_Accuracy: {np.mean(val_acc_per_fold)*100} (+- {np.std(val_acc_per_fold)})')