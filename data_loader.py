# load, split and scale the maps dataset ready for training
from os import listdir

import numpy as np
from tensorflow.keras.preprocessing.image import *


class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        if is_testing:
            covers_path = './test_data/'
        else:
            covers_path = './train_data/covers/'
        masks_path = './train_data/masks/'

        batch_images = list(filter(lambda f: not f.startswith('.'), sorted(listdir(covers_path))))[:batch_size]

        covers, masks = list(), list()
        for img_path in batch_images:
            cover = load_img(covers_path + img_path, target_size=self.img_res, color_mode='grayscale')
            mask = load_img(masks_path + img_path, target_size=self.img_res)
            cover = img_to_array(cover)
            mask = img_to_array(mask)

            if not is_testing and np.random.random() > 0.5:
                cover = np.fliplr(cover)
                mask = np.fliplr(mask)

            covers.append(cover)
            masks.append(mask)

        covers = np.array(covers) / 127.5 - 1.
        masks = np.array(masks) / 127.5 - 1.

        return covers, masks

    def load_batch(self, batch_size=1, is_testing=False):
        if is_testing:
            covers_path = './test_data/'
        else:
            covers_path = './train_data/covers/'
        masks_path = './train_data/masks/'

        self.n_batches = int(len(listdir(covers_path)) / batch_size)

        data = list(filter(lambda f: not f.startswith('.'), sorted(listdir(covers_path))))

        for i in range(self.n_batches - 1):
            batch = data[i * batch_size:(i + 1) * batch_size]
            print(batch)
            covers, masks = list(), list()
            for img in batch:
                cover = load_img(covers_path + img, target_size=self.img_res, color_mode='grayscale')
                mask = load_img(masks_path + img, target_size=self.img_res)
                cover = img_to_array(cover)
                mask = img_to_array(mask)

                if not is_testing and np.random.random() > 0.5:
                    cover = np.fliplr(cover)
                    mask = np.fliplr(mask)

                covers.append(cover)
                masks.append(mask)

            covers = np.array(covers) / 127.5 - 1.
            masks = np.array(masks) / 127.5 - 1.

            yield covers, masks
