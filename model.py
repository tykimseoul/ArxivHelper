from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
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
    def __init__(self, num_class, input_size=(512, 512, 1), batch_size=1):
        self.base_model = None
        self.input_size = input_size
        self.num_class = num_class
        self.batch_size = batch_size
        self.model = self.build()

    def build(self):
        def conv_block(inputs, filters, kernel_size, batch_normalization=True, residual=True):
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            if residual:
                shortcut = Conv2D(filters, 1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
                shortcut = BatchNormalization()(shortcut)
                conv = Add()([shortcut, conv])
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
        title_other_channel = Concatenate(axis=1)([title_other_channel, Ones()(shape=(self.batch_size,) + title_other_channel.shape[1:])])

        abstract_other_channel, abstract_channel = extract_channels(abstract_block)

        other_channel = Multiply()([title_other_channel, abstract_other_channel])

        output = Concatenate()([title_channel, other_channel, abstract_channel])

        assert output.shape[3] == self.num_class

        model = Model(inputs, output)

        model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[tversky, 'accuracy'])

        model.summary()

        return model
