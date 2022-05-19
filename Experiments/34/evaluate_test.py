import tensorflow as tf
import os
import sys
import inspect

model = tf.keras.models.load_model('best_model')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)
import DataGenerator
import ModelEvaluator

img_width, img_height = 256, 256
batch_size = 64

_, _, test_generator = DataGenerator.data_Gens(parentparentdir, img_height, img_width, batch_size)
print('data loaded')

ModelEvaluator.evaluate_test(model, test_generator)