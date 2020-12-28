import os
import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import Unet
from data_loader import DataLoader


class Pix2Pix:
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.cover_shape = (self.img_rows, self.img_cols, 1)
        self.mask_shape = (self.img_rows, self.img_cols, 3)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = tf.keras.layers.Input(shape=self.mask_shape)
        img_B = tf.keras.layers.Input(shape=self.cover_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = tf.keras.models.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        unet = Unet(input_size=(256, 256, 1), num_class=3)
        return unet.model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            if bn:
                d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
            return d

        img_A = tf.keras.layers.Input(shape=self.mask_shape)
        img_B = tf.keras.layers.Input(shape=self.cover_shape)
        print(img_A.shape, img_B.shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        model = tf.keras.models.Model([img_A, img_B], validity)

        model.summary()

        return model

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (cover, mask) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_mask = self.generator.predict(cover)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([mask, cover], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_mask, cover], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([mask, cover], [valid, mask])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print('[Epoch {}/{}] [Batch {}/{}] [D {}: {}, {}: {}] [G {}: {}] time: {}'.format(epoch, epochs,
                                                                                                  batch_i, self.data_loader.n_batches,
                                                                                                  self.discriminator.loss, d_loss[0], self.discriminator.metrics_names, 100 * d_loss[1],
                                                                                                  self.combined.loss, g_loss[0],
                                                                                                  elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('./images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 5

        cover, mask = self.data_loader.load_data(batch_size=c, is_testing=True)
        fake_mask = self.generator.predict(cover)

        cover = np.tile(cover, (1, 1, 1, 3))

        print(cover.shape)
        gen_imgs = np.concatenate([cover, fake_mask, mask])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.show()
        plt.close()


gan = Pix2Pix()
gan.train(epochs=10, batch_size=1, sample_interval=200)
