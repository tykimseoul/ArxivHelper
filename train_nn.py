from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import numpy as np

import matplotlib.pyplot as plt


def custom_loss(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    return K.sum(diff)


def custom_metric(y_true, y_pred):
    diff = K.abs(y_true - y_pred)
    return -K.sum(diff)


def train_data_generator(batch_size, image_path, mask_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                         mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                         save_to_dir=None, target_size=(512, 512), seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    train_datagen = ImageDataGenerator(**aug_dict)
    image_train_generator = train_datagen.flow_from_directory(
        image_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        subset='training')
    image_valid_generator = train_datagen.flow_from_directory(
        image_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
        subset='validation')
    mask_train_generator = train_datagen.flow_from_directory(
        mask_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        subset='training')
    mask_valid_generator = train_datagen.flow_from_directory(
        mask_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
        subset='validation')
    return image_train_generator, mask_train_generator, image_valid_generator, mask_valid_generator

def yield_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        area = get_abstract_area(mask)
        yield img, area


def get_abstract_area(mask):
    mask = mask[0, :, :, :]
    # print(mask.shape)
    # plt.figure()
    # plt.imshow(mask)
    # plt.show()
    coordinates = np.where(np.all(mask == [0, 0, 255], axis=-1))
    coordinates = np.array(coordinates)
    # print(coordinates, coordinates.shape)
    x_min = np.amin(coordinates[1, :]) / (mask.shape[1] - 1)
    x_max = np.amax(coordinates[1, :]) / (mask.shape[1] - 1)
    y_min = np.amin(coordinates[0, :]) / (mask.shape[0] - 1)
    y_max = np.amax(coordinates[0, :]) / (mask.shape[0] - 1)
    # print(x_min, x_max, y_min, y_max)
    return np.array([x_min, x_max, y_min, y_max])


class ArxivModel:
    def __init__(self, input_size=(512, 512, 1)):
        self.input_size = input_size
        self.model = self.build()

    def build(self):
        inputs = Input(self.input_size)

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        flat = Flatten()(pool4)

        dense = Dense(512, activation='relu')(flat)
        dense = Dense(128, activation='relu')(dense)
        dense = Dense(64, activation='relu')(dense)
        dense = Dense(4, activation='sigmoid')(dense)

        model = Model(inputs, dense)
        model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=custom_metric)

        model.summary()

        return model


if __name__ == "__main__":
    data_gen_args = dict(width_shift_range=0.05,
                         height_shift_range=0.05,
                         zoom_range=0.05,
                         validation_split=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
    generator = train_data_generator(1, '/kaggle/input/dataset3/train_unet', '/kaggle/input/dataset3/masks_unet', 'train_unet', 'masks_unet', data_gen_args, image_color_mode="grayscale", mask_color_mode="rgb", save_to_dir=None)
    train_generator = yield_generator(generator[0], generator[1])
    valid_generator = yield_generator(generator[2], generator[3])
    model_checkpoint = ModelCheckpoint('nn_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model = ArxivModel()
    history = model.model.fit(train_generator, validation_data=valid_generator, validation_steps=200, steps_per_epoch=2000, epochs=15, callbacks=[model_checkpoint])