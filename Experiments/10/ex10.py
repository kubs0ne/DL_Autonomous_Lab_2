import os
import sys
import inspect
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
from keras import regularizers
from tensorflow.keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn import svm
import random

num_classes = 29
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator

img_width, img_height = 256, 256
batch_size = 64
epochs = 10

x_train, x_val, x_test = DataGenerator.load_mame(parentparentdir,  dataframe=False)

model= tf.keras.applications.DenseNet121(include_top=False, input_shape=(img_width,img_width,3), weights="imagenet", pooling = 'avg')

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in model.layers[:10]:
    layer.trainable = False

def extract_features(data, model):
    X_list = []
    Y_list = []
    for path, label in zip(data[0],data[1]):
        # load image
        img = image.load_img(path, target_size=(256,256))
        # convert image to numpy array
        img_arr = image.img_to_array(img)
        # add 1 more dimension
        img_arr_b = np.expand_dims(img_arr, axis=0)
        # preprocess image
        input_img = applications.densenet.preprocess_input(img_arr_b)
        # extract feature
        feature_vec = model.predict(input_img)

        X_list.append(feature_vec.ravel())
        Y_list.append(label)
    return X_list,Y_list

X_train,Y_train = extract_features(x_train, model)
X_test,Y_test = extract_features(x_test, model)
print(X_train)
print(Y_train)

svm_lin = svm.SVC(C=1.0, kernel="linear")
svm_lin.fit(X_train, Y_train)
y_pred = svm_lin.predict(X_test)
print(classification_report(Y_test, y_pred))

