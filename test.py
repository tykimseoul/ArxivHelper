import os
import cv2
from train import *
from PIL import Image

title = [255, 0, 0]
author = [0, 255, 0]
abstract = [0, 0, 255]
other = [0, 0, 0]
COLOR_DICT = np.array([title, author, abstract, other])


def test_generator(test_path, target_size=(512, 512)):
    for i in sorted(os.listdir(test_path))[:10]:
        print(i)
        img = cv2.imread(test_path + i, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        img = np.reshape(img, (1,) + img.shape)
        yield img


def label_visualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out


def save_result(npyfile, num_class):
    print(npyfile.shape)
    for i, item in enumerate(npyfile):
        img = label_visualize(num_class, COLOR_DICT, item)
        img = Image.fromarray(img.astype(np.uint8))
        img.save('./results/result_{}.png'.format(i))


testGene = test_generator('./test_data/covers/')
model = Unet(4)
model.load_weights("./unet_membrane.hdf5")
results = model.predict(testGene, 10, verbose=1)
save_result(results, 4)
