import os
import sys
import inspect
import time
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
from keras import regularizers
from tensorflow.keras import applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import optimizers

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


# x_train, y_train, x_val, y_val, x_test, y_test = DataGenerator.load_mame(parentparentdir,  dataframe=False)
x_train, x_val, x_test = DataGenerator.load_mame(parentparentdir,  dataframe=False)
print(x_train[0])
print(x_train[1])
#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers

model= tf.keras.applications.DenseNet121(include_top=False, input_shape=(img_width,img_width,3), weights="imagenet")

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in model.layers[:10]:
    layer.trainable = False
# model.summary()
#
# pred = model.predict()
#
#
# print('Model trained in {:.1f}min'.format((time.time() - t0) / 60))
#
# ModelEvaluator.evaluate_model(model_final, history, validation_generator)
