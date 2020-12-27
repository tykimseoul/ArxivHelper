import os
import cv2
from train import *
from PIL import Image


def test_generator(test_path, target_size=(512, 512)):
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

    for i in sorted(os.listdir(test_path)):
        if i.startswith('.'):
            continue
        print(i)
        img = cv2.imread(test_path + i, cv2.IMREAD_GRAYSCALE)
        img = img / 255
        img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        boundary = delineate(img)
        img = np.expand_dims(img, 2)
        boundary = np.expand_dims(boundary, 2)
        img = np.expand_dims(img, 0)
        boundary = np.expand_dims(boundary, 0)
        # yield np.squeeze(np.stack((img, boundary), axis=3), axis=4)
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
    npyfile=np.array(npyfile)
    npyfile=npyfile[3, :]
    print(npyfile.shape)
    for i, item in enumerate(npyfile):
        print(i, item.shape)
        img = label_visualize(item)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((612, 792))
        img.save('./results/result_{}_3.png'.format(i))


if __name__ == "__main__":
    testGene = test_generator('./test_data/')
    num_class = 3
    model = Unet(num_class, deep_supervision=True)
    color_map = np.eye(num_class)
    model.model.load_weights('./unet_membrane_nested_deep.hdf5')
    results = model.model.predict(testGene, verbose=1)
    save_result(results)
