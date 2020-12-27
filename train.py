from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import ndimage
from model import Unet


def adjust_data(img, mask, num_class):
    def delineate(img):
        def sigmoid(x, c):
            return 1 / (1 + np.exp(-c * x))

        def sobel(img, size):
            img = ndimage.maximum_filter(img, size=size)
            img = sigmoid(img, 5)
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
        img, boundary, mask = adjust_data(img, mask, num_class)
        # print(img.shape, mask.shape)
#         yield np.squeeze(np.stack((img, boundary), axis=3), axis=4), mask
        yield img, mask


if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         validation_split=0.2,
                         fill_mode='nearest')
    num_class = 3
    title = [255, 0, 0]
    abstract = [0, 0, 255]
    other = [0, 0, 0]
    color_map = np.array([title, other, abstract][:num_class])
    generator = train_data_generator(1, '/kaggle/input/dataset/train_unet', '/kaggle/input/dataset/masks_unet', 'train_unet', 'masks_unet', data_gen_args, image_color_mode="grayscale", mask_color_mode="rgb", save_to_dir=None)
    train_generator = yield_generator(generator[0], generator[1])
    valid_generator = yield_generator(generator[2], generator[3])
    model = Unet(num_class, deep_supervision=True)
    model_checkpoint = ModelCheckpoint('/kaggle/working/unet_membrane_title.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # history = model.model.fit(train_generator, validation_data=valid_generator, validation_steps=200, steps_per_epoch=2000, epochs=20, callbacks=[model_checkpoint])
