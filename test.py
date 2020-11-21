import os
import cv2
from train import *
from PIL import Image

title = [1, 0, 0, 0]
author = [0, 1, 0, 0]
abstract = [0, 0, 1, 0]
color_map = np.array([title, author, abstract])


def test_generator(test_path, target_size=(512, 512)):
    for i in sorted(os.listdir(test_path)):
        print(i)
        img = cv2.imread(test_path + i, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        img = np.reshape(img, (1,) + img.shape)
        yield img


def label_visualize(img):
    reshaped_img = np.reshape(img, [img.shape[0] * img.shape[1], img.shape[2]])
    maxed_img = np.zeros_like(reshaped_img)
    maxed_img[np.arange(len(reshaped_img)), reshaped_img.argmax(axis=1)] = 1
    img_out = np.zeros(maxed_img.shape[:1] + (3,))
    for i, color in enumerate(color_map):
        img_out[np.all(maxed_img == color, axis=-1), i] = 255
    img_out = np.reshape(img_out, (img.shape[:2] + (3,)))
    return img_out


def save_result(npyfile):
    print(npyfile.shape)
    for i, item in enumerate(npyfile):
        img = label_visualize(item)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((612, 792))
        img.save('./results/result_{}.png'.format(i))


testGene = test_generator('./test_data/covers/')
model = Unet(4)
model.load_weights("./unet_membrane.hdf5")
results = model.predict(testGene, 10, verbose=1)
save_result(results)
