import tensorflow as tf
import sys

num = sys.argv[1]
model = tf.keras.models.load_model(f'Experiments/{num}/best_model')

model_json = model.to_json()
with open(f'Models/model{num}.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(f'Models/model{num}.h5')
print("Saved model to disk")