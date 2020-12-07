from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import *
import numpy as np
from scipy import ndimage
import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


class Unet:
    def __init__(self, num_class, input_size=(512, 512, 1)):
        self.base_model = None
        self.input_size = input_size
        self.num_class = num_class
        self.model = self.build()

    def build(self):
        def conv_block(inputs, filters, kernel_size, batch_normalization=True):
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            return conv

        inputs = Input(self.input_size)

        conv1 = conv_block(inputs, 32, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = conv_block(pool1, 64, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = conv_block(pool2, 128, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = conv_block(pool3, 256, 3)

        def title_branch(input):
            crop1 = Cropping2D(cropping=((0, int(input.shape[1] * 0.5)), (0, 0)))(input)
            pool4 = MaxPooling2D(pool_size=(2, 2))(crop1)
            conv5 = conv_block(pool4, 512, 3)

            up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
            merge6 = Concatenate()([Cropping2D(cropping=((0, int(conv4.shape[1] * 0.5)), (0, 0)))(conv4), up6])
            conv6 = conv_block(merge6, 256, 3)

            up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
            merge7 = Concatenate()([Cropping2D(cropping=((0, int(conv3.shape[1] * 0.5)), (0, 0)))(conv3), up7])
            conv7 = conv_block(merge7, 128, 3)

            up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
            merge8 = Concatenate()([Cropping2D(cropping=((0, int(conv2.shape[1] * 0.5)), (0, 0)))(conv2), up8])
            conv8 = conv_block(merge8, 64, 3)

            up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
            merge9 = Concatenate()([Cropping2D(cropping=((0, int(conv1.shape[1] * 0.5)), (0, 0)))(conv1), up9])
            conv9 = conv_block(merge9, 32, 3)

            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = BatchNormalization()(conv9)
            conv10 = Conv2D(2, 1, activation='softmax')(conv9)

            return conv10

        def abstract_branch(input):
            pool4 = MaxPooling2D(pool_size=(2, 2))(input)

            conv5 = conv_block(pool4, 512, 3)

            up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
            merge6 = Concatenate()([conv4, up6])
            conv6 = conv_block(merge6, 256, 3)

            up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
            merge7 = Concatenate()([conv3, up7])
            conv7 = conv_block(merge7, 128, 3)

            up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
            merge8 = Concatenate()([conv2, up8])
            conv8 = conv_block(merge8, 64, 3)

            up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
            merge9 = Concatenate()([conv1, up9])
            conv9 = conv_block(merge9, 32, 3)

            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = BatchNormalization()(conv9)
            conv10 = Conv2D(2, 1, activation='softmax')(conv9)

            return conv10

        title_block = title_branch(conv4)
        abstract_block = abstract_branch(conv4)

        def extract_channels(input):
            input = K.expand_dims(input, axis=-1)
            first = Cropping3D(cropping=((0, 0), (0, 0), (0, 1)))(input)
            second = Cropping3D(cropping=((0, 0), (0, 0), (1, 0)))(input)
            first = K.squeeze(first, axis=-1)
            second = K.squeeze(second, axis=-1)
            return first, second

        title_channel, title_other_channel = extract_channels(title_block)
        title_channel = ZeroPadding2D(padding=((0, title_channel.shape[1]), (0, 0)))(title_channel)
        title_other_channel = Concatenate(axis=1)([title_other_channel, Ones()(shape=(batch_size,) + title_other_channel.shape[1:])])

        abstract_other_channel, abstract_channel = extract_channels(abstract_block)

        other_channel = Multiply()([title_other_channel, abstract_other_channel])

        output = Concatenate()([title_channel, other_channel, abstract_channel])

        assert output.shape[3] == self.num_class

        model = Model(inputs, output)

        model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[tversky, 'accuracy'])

        model.summary()

        return model


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


def train_data_generator(batch_size, image_path, mask_path, image_folder, mask_folder, aug_dict, num_class, image_color_mode="grayscale",
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
    batch_size = 1
    num_class = 3
    title = [255, 0, 0]
    abstract = [0, 0, 255]
    other = [0, 0, 0]
    color_map = np.array([title, other, abstract][:num_class])
    generator = train_data_generator(batch_size, '/kaggle/input/dataset/train_unet', '/kaggle/input/dataset/masks_unet', 'train_unet', 'masks_unet', data_gen_args, num_class, image_color_mode="grayscale", mask_color_mode="rgb", save_to_dir=None)
    train_generator = yield_generator(generator[0], generator[1])
    valid_generator = yield_generator(generator[2], generator[3])
    model = Unet(num_class)
    model_checkpoint = ModelCheckpoint('/kaggle/working/unet_membrane_title.hdf5', monitor='loss', verbose=1, save_best_only=True)
    history = model.model.fit(train_generator, validation_data=valid_generator, validation_steps=200, steps_per_epoch=2000, epochs=20, callbacks=[model_checkpoint])
