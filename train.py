from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import ndimage


class Unet:
    def __init__(self, num_class, input_size=(400, 400, 2)):
        self.base_model = None
        self.input_size = input_size
        self.num_class = num_class
        self.model = self.build()

    def build(self):
        inputs = Input(self.input_size)

        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(self.num_class, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(self.num_class, 1, activation='softmax')(conv9)

        model = Model(inputs, conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics='accuracy')

        model.summary()

        return model


def adjust_data(img, mask, num_class):
    def delineate(img):
        def sobel(img, size):
            img = ndimage.maximum_filter(img, size=size)
            img = np.around(img)
            sx = ndimage.sobel(img, axis=0, mode='constant')
            sy = ndimage.sobel(img, axis=1, mode='constant')
            sobel = np.hypot(sx, sy)
            gaussian = ndimage.gaussian_filter(sobel, sigma=3)
            return gaussian

        img = 1 - img

        sobel1 = sobel(img, size=10)
        sobel2 = sobel(img, size=16)
        sobel3 = sobel(img, size=22)

        avg = np.mean([sobel1, sobel2, sobel3], axis=0)
        avg /= np.amax(avg)

        return img

    img = img / 255
    boundary = delineate(img)
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, 0)
    reshaped_mask = np.reshape(mask, [mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3]])
    new_mask = np.zeros(reshaped_mask.shape[:2] + (num_class,))
    for b in range(reshaped_mask.shape[0]):
        # for each batch
        for i, color in enumerate(color_map):
            new_mask[b, np.all(reshaped_mask[b] == color, axis=-1), i] = 1
    new_mask = np.reshape(new_mask, (mask.shape[0], mask.shape[1], mask.shape[2], new_mask.shape[2]))
    mask = new_mask
    return img, boundary, mask


def train_data_generator(batch_size, image_path, mask_path, image_folder, mask_folder, aug_dict, num_class, image_color_mode="grayscale",
                         mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                         save_to_dir=None, target_size=(400, 400), seed=1):
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
        img, boundary, mask = adjust_data(img, mask, num_class)
        print(img.shape, boundary.shape, mask.shape)
        print(np.stack((img, boundary), axis=3).shape)
        yield np.squeeze(np.stack((img, boundary), axis=3), axis=4), mask


if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         validation_split=0.2,
                         fill_mode='nearest')
    num_class = 2
    title = [255, 0, 0]
    author = [0, 255, 0]
    abstract = [0, 0, 255]
    other = [0, 0, 0]
    color_map = np.array([other, abstract, title, author][:num_class])
    generator = train_data_generator(1, '/kaggle/input/dataset/train_unet', '/kaggle/input/dataset/masks_unet', 'train_unet', 'masks_unet', data_gen_args, num_class, image_color_mode="grayscale", mask_color_mode="rgb", save_to_dir=None)
    train_generator = yield_generator(generator[0], generator[1])
    valid_generator = yield_generator(generator[2], generator[3])
    model = Unet(num_class)
    model_checkpoint = ModelCheckpoint('/kaggle/working/unet_membrane_title.hdf5', monitor='loss', verbose=1, save_best_only=True)
    history = model.model.fit(train_generator, validation_data=valid_generator, validation_steps=2000, steps_per_epoch=2000, epochs=20, callbacks=[model_checkpoint])
