import os
import sys
import inspect
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
import numpy as np
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

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

model= tf.keras.applications.ResNet50(include_top=False, input_shape=(img_width,img_width,3), weights="imagenet")

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
for layer in model.layers:
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
        input_img = applications.resnet50.preprocess_input(img_arr_b)
        # extract feature
        feature_vec = model.predict(input_img)

        X_list.append(feature_vec.ravel())
        Y_list.append(label)
    return X_list,Y_list

X_train,Y_train = extract_features(x_train, model)
X_test,Y_test = extract_features(x_test, model)
print(X_train)
print(Y_train)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, verbose=1)
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
y_pred = clf.predict(X_test)
print('accuracy: ',score)
print(confusion_matrix(Y_test, y_pred))


# svm_lin = svm.SVC(C=1.0, kernel="linear")
# svm_lin.fit(X_train, Y_train)
# y_pred = svm_lin.predict(X_test)
# print(classification_report(Y_test, y_pred))

