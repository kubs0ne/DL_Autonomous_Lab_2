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
batch_size = 124
epochs = 120

train_generator, validation_generator, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)


#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers

model= tf.keras.applications.DenseNet121(include_top=False, input_shape=(img_width,img_width,3), weights="imagenet")

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(units = 1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense( 
        units = 512, 
        activation="relu")(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense( 
        units = 256, 
        activation="relu")(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
predictions = Dense(num_classes, activation="softmax")(x)


# creating the final model
model_final = Model(model.input, predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()

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




history = model_final.fit_generator(
    generator=train_generator,
    steps_per_epoch= STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps= STEP_SIZE_VAL,
    epochs=epochs,
    callbacks=[earlyStopping, mcp_save_best]
)

ModelEvaluator.evaluate_model(model_final, history, validation_generator)
