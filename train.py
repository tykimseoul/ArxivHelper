from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

title = [255, 0, 0]
author = [0, 255, 0]
abstract = [0, 0, 255]
other = [0, 0, 0]
color_map = np.array([title, author, abstract, other])


def cast_f(x):
    return K.cast(x, K.floatx())


def cast_b(x):
    return K.cast(x, bool)


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    not_true = 1 - true
    union = true + (not_true * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def iou_metric(true, pred):  # any shape can go - can't be a loss function

    thresholds = [0.5 + (i * .05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = cast_f(K.greater(pred, 0.5))

    # total white pixels - (batch,)
    true_sum = K.sum(true, axis=-1)
    pred_sum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = cast_f(K.greater(true_sum, 1))
    pred1 = cast_f(K.greater(pred_sum, 1))

    # to get images that have mask in both true and pred
    true_positive_mask = cast_b(true1 * pred1)

    # separating only the possible true positives to check iou
    test_true = tf.boolean_mask(true, true_positive_mask)
    test_pred = tf.boolean_mask(pred, true_positive_mask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(test_true, test_pred)
    true_positives = [cast_f(K.greater(iou, tres)) for tres in thresholds]

    # mean of thresholds for true positives and total sum
    true_positives = K.mean(K.stack(true_positives, axis=-1), axis=-1)
    true_positives = K.sum(true_positives)

    # to get images that don't have mask in both true and pred
    true_negatives = (1 - true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    true_negatives = K.sum(true_negatives)

    return (true_positives + true_negatives) / cast_f(K.shape(true)[0])


def IoULoss(y_true, y_pred, smooth=0):
    # flatten label and prediction tensors

    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    print(y_pred.shape)
    print(y_true.shape)

    intersection = K.sum(K.batch_dot(y_true, y_pred), keepdims=False)
    print('intx', intersection.shape)
    total = K.sum(y_true, keepdims=False) + K.sum(y_pred, keepdims=False)
    print('total', total.shape)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def IoU(y_true, y_pred, smooth=0):
    # flatten label and prediction tensors

    y_pred = K.batch_flatten(y_pred)
    y_true = K.batch_flatten(y_true)
    print(y_pred.shape)
    print(y_true.shape)

    intersection = K.sum(K.batch_dot(y_true, y_pred), keepdims=False)
    print('intx', intersection.shape)
    total = K.sum(y_true, keepdims=False) + K.sum(y_pred, keepdims=False)
    print('total', total.shape)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return IoU


class Unet:
    def __init__(self, num_class, input_size=(512, 512, 1)):
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

    def fit(self, data, steps_per_epoch, epochs, callbacks):
        self.model.fit_generator(data, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

    def predict(self, test_gene, batch_size, verbose):
        return self.model.predict(test_gene, batch_size, verbose=verbose)

    def load_weights(self, weights):
        self.model.load_weights(weights)


def adjust_data(img, mask, num_class):
    img = img / 255
    if len(mask.shape) == 3:
        mask = np.expand_dims(mask, 0)
    reshaped_mask = np.reshape(mask, [mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3]])
    new_mask = np.zeros(reshaped_mask.shape[:2] + (num_class,))
    for b in range(reshaped_mask.shape[0]):
        # for each batch
        for i, color in enumerate(color_map):
            new_mask[b, np.all(reshaped_mask[b] == color, axis=-1), i] = 1
    new_mask = np.reshape(new_mask, (mask.shape[0], mask.shape[1], mask.shape[2], new_mask.shape[2]))
    # print(img.shape, mask.shape)
    return img, new_mask


def train_data_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                         mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask", num_class=4,
                         save_to_dir=None, target_size=(512, 512), seed=1):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, num_class)
        yield img, mask


if __name__ == "__main__":
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    myGene = train_data_generator(2, './train_data', 'covers', 'masks', data_gen_args, image_color_mode="grayscale", mask_color_mode="rgb", save_to_dir=None)
    model = Unet(4)
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(myGene, steps_per_epoch=2000, epochs=5, callbacks=[model_checkpoint])
