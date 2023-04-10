import os.path
import random
from typing import Iterator

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
import cv2
import numpy as np
from PIL import ImageChops, Image, ImageDraw
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import cm, gridspec
from pycocotools.coco import COCO
from definitions import DATASET_PATH, ROOT_DIR

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getNormalMask(coco, image_id, catIds, input_image_size, classes):
    # print(image_id)
    annIds = coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    # print(anns)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size, dtype=np.uint8)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = catIds
        # print(pixel_value)
        new_mask = cv2.resize(coco.annToMask(
            anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    return train_mask


def getLevelsMask(coco, image_id, catIds, input_image_size):
    # for each category , we get the x mask and add it to mask list
    res = []
    for j, categorie in enumerate(catIds):
        mask = getNormalMask(coco=coco, image_id=image_id, catIds=categorie, input_image_size=input_image_size,
                             classes=catIds)
        res.append(mask)
    return res


def getImage(file_path, input_image_size):
    train_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    train_img = cv2.resize(train_img, (input_image_size))
    train_img = train_img.astype(np.float32) / 255.
    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):
        return train_img
    else:
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getImagePathByCocoId(coco, image_id):
    image = coco.loadImgs([image_id])[0]
    imagePath = DATASET_PATH + 'train/' + image['file_name']
    # if path_folder_image is not None:
    #     pass
    #     imagePath = f'{DATASET_PATH}/{path_folder_image}/{image["file_name"]}'

    # print('folder is not none')
    # print(imagePath)
    # print(imagePath)
    return imagePath

def cocoDataGenerator(images, classes, coco, folder=None,
                      input_image_size=(224, 224), batch_size=2, mode='train',
                      mask_type='binary',
                      shuffle=False):
    # img_folder = '{}/images/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    c = 0
    if shuffle:
        random.shuffle(images)
    while (True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], len(classes))).astype('float')
        # print(img.shape)
        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            file_path_image = getImagePathByCocoId(coco, image_id=imageObj["id"])
            # print(file_path_image)
            train_img = getImage(file_path=file_path_image, input_image_size=input_image_size)
            mask_train = getLevelsMask(coco, imageObj['id'], catIds, input_image_size)
            # print(np.array(mask_train).shape)
            for j in range(len(classes)):
                mask[i-c, :, :, j-1] = mask_train[j-1]
            img[i - c] = train_img

            # plt.imshow(img[i-c])
            # plt.show()

        c += batch_size
        if (c + batch_size >= dataset_size):
            c = 0
            # print('WOOOOOOOOOOOOOOORK')
            random.shuffle(images)

        yield img, mask


def visualizeGenerator(gen, img=None, pred=None):
    import matplotlib as mpl
    colors = ['#0044ff', '#ff00fb', '#ff0000', '#2bff00', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    # Iterate the generator to get image and mask batches
    if gen is not None:
        img1, mask1 = next(gen)
    else:
        img1, mask1 = img, pred
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)
        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img1[j])
            else:
                # print(mask.shape)
                for m in range(len(mask1[0, 0, 0, :])):
                    mask_one = mask1[j, :, :, m]
                    if pred is not None:
                        mask_one = mask1[j, :, :, m] > 0.85
                    ax.imshow(mask_one, alpha=0.5)
            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()


