import pandas as pd
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(2020)


def load_mame(path, dataframe=False):
    INPUT_PATH = 'MAMe_metadata'
    # Load dataset table with descriptions
    dataset = pd.read_csv(os.path.join(path+os.sep,INPUT_PATH+os.sep, 'MAMe_dataset.csv'))

    # Subset divisions
    x_train_files = dataset.loc[dataset['Subset'] == 'train']['Image file'].tolist()
    y_train_class = dataset.loc[dataset['Subset'] == 'train']['Medium'].tolist()

    x_val_files = dataset.loc[dataset['Subset'] == 'val']['Image file'].tolist()
    y_val_class = dataset.loc[dataset['Subset'] == 'val']['Medium'].tolist()

    x_test_files = dataset.loc[dataset['Subset'] == 'test']['Image file'].tolist()
    y_test_class = dataset.loc[dataset['Subset'] == 'test']['Medium'].tolist()

    if dataframe:
        train = pd.DataFrame({'filename': x_train_files, 'class': y_train_class})
        val = pd.DataFrame({'filename': x_val_files, 'class': y_val_class})
        test = pd.DataFrame({'filename': x_test_files, 'class': y_test_class})

        # Set full path
        train['filename'] = train['filename'].transform(
            lambda x: path + os.sep + INPUT_PATH + os.sep + 'data' + os.sep + x)
        val['filename'] = val['filename'].transform(
            lambda x: path + os.sep + INPUT_PATH + os.sep + 'data' + os.sep + x)
        test['filename'] = test['filename'].transform(
            lambda x: path + os.sep + INPUT_PATH+ os.sep + 'data' + os.sep + x)

        return train, val, test

    else:
        # Return list of filenames
        x_train = [os.path.join(path, INPUT_PATH, 'data', img_name) for img_name in x_train_files]
        x_val = [os.path.join(path, INPUT_PATH, 'data', img_name) for img_name in x_val_files]
        x_test = [os.path.join(path, INPUT_PATH, 'data', img_name) for img_name in x_test_files]

        return (np.array(x_train), np.array(y_train_class)), (np.array(x_val),
                                                              np.array(y_val_class)), (
               np.array(x_test), np.array(y_test_class))

def data_Gens(path, img_height, img_width, batch_size):
    df_train, df_val, df_test = load_mame(path,dataframe=True)
    # Initiate the train and test generators
    train_datagen = ImageDataGenerator()

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
            df_train,
            target_size = (img_height, img_width),
            batch_size = batch_size,
            class_mode = "categorical",
            validate_filenames=False)

    validation_generator = test_datagen.flow_from_dataframe(
            df_val,
            target_size = (img_height, img_width),
            batch_size = batch_size,
            shuffle = False,
            class_mode = "categorical",
            validate_filenames=False)

    test_generator = test_datagen.flow_from_dataframe(
            df_test,
            target_size = (img_height, img_width),
            batch_size = 1,
            shuffle = False,
            class_mode = "categorical",
            validate_filenames=False)

    return train_generator, validation_generator, test_generator