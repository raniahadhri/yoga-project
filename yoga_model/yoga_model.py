# Function(s) for loading and using the model
##the model 

import tensorflow.keras.layers as tfl
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import model_utils as d
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


#-------------------------------------------------------------------------------------------------
#define the model
model = d.build_model(class_num=class_num)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True,random_state=2)

# First training before data augmentation
val_acc_per_fold,val_loss_per_fold,loss_per_fold,acc_per_fold,model_1=d.training_model_1(train_images_data, train_images_label, model, kfold)
print(f'> Mean_Training_Accuracy: {np.mean(acc_per_fold)*100} (+- {np.std(acc_per_fold)})')
print(f'> Mean_Validation_Accuracy: {np.mean(val_acc_per_fold)*100} (+- {np.std(val_acc_per_fold)})')

# Second training after data augmentation
dataAugmentation= d.dataAugmentation()
val_acc_per_fold,val_loss_per_fold,loss_per_fold,acc_per_fold, model_2=d.training_model_2(train_images_data, train_images_label, model, kfold,dataAugmentation)
print(f'> Mean_Training_Accuracy: {np.mean(acc_per_fold)*100} (+- {np.std(acc_per_fold)})')
print(f'> Mean_Validation_Accuracy: {np.mean(val_acc_per_fold)*100} (+- {np.std(val_acc_per_fold)})')

#------------------------------------------------------------------------------------------------------
model_2.save('saved_models/yoga_pose_classifier.h5')
