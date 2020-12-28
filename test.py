import os
import cv2
from train import *
from PIL import Image
from tensorflow.keras.preprocessing.image import *


def test_generator(test_path, target_size=(512, 512)):

    for i in sorted(os.listdir(test_path)):
        if i.startswith('.'):
            continue
        img = load_img(test_path + i, target_size=target_size, color_mode='grayscale')
        img = np.array(img)
        img = img / 255
        img = np.expand_dims(img, 2)
        img = np.expand_dims(img, 0)
        yield img


def label_visualize(img):
    reshaped_img = np.reshape(img, [img.shape[0] * img.shape[1], img.shape[2]])
    maxed_img = np.zeros_like(reshaped_img)
    maxed_img[np.arange(len(reshaped_img)), reshaped_img.argmax(axis=1)] = 1
    img_out = np.zeros(maxed_img.shape[:1] + (3,))
    for i, color in enumerate(color_map):
        img_out[np.where(np.all(maxed_img == color, axis=-1)), i] = 255
    img_out = np.reshape(img_out, (img.shape[:2] + (3,)))
    return img_out


def save_result(npyfile):
    print(npyfile.shape)
    for i, item in enumerate(npyfile):
        print(i, item.shape)
        # img = label_visualize(item)
        item = item * 255
        img = Image.fromarray(item.astype(np.uint8))
        img = img.resize((612, 792))
        img.save('./results/result_{}.png'.format(i))


if __name__ == "__main__":
    testGene = test_generator('./test_data/', target_size=(256, 256, 1))
    num_class = 3
    model = Unet(num_class, input_size=(256, 256, 1), deep_supervision=False)
    color_map = np.eye(num_class)
    model.model.load_weights('./gan.hdf5')
    results = model.model.predict(testGene, verbose=1)
    save_result(results)
