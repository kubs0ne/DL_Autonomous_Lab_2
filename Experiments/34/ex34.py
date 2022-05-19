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
import keras

num_classes = 29
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator


img_width, img_height = 256, 256
batch_size = 64
epochs = 120

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)


#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers

model = keras.models.load_model('/home/nct01/nct01036/AutoLab2/Models/model_ex_29')
# mark loaded layers as not trainable
for i, layer in enumerate(model.layers):
    layer.trainable = True


# compile the model
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model.summary()

# Train the model
t0 = time.time()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = validation_generator.n // validation_generator.batch_size

early = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto',
                      restore_best_weights=True)

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# Save the best model and also safe the last model which is not more 1% discrepance between validation accu
mcp_save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)




history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch= STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps= STEP_SIZE_VAL,
    epochs=epochs,
    callbacks=[earlyStopping, mcp_save_best]
)

ModelEvaluator.evaluate_model(model, history, validation_generator)
