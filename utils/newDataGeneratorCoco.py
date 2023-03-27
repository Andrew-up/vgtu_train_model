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


def getBinaryMask(_, o, j, s):
    return 0


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
    imagePath = DATASET_PATH + '/' + image['file_name']
    # if path_folder_image is not None:
    #     pass
    #     imagePath = f'{DATASET_PATH}/{path_folder_image}/{image["file_name"]}'

    # print('folder is not none')
    # print(imagePath)
    # print(imagePath)
    return imagePath


def hui():
    return 'хуй блять'


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


def visualizeGenerator(gen):
    import matplotlib as mpl
    colors = ['#0044ff', '#ff00fb', '#ff0000', '#2bff00', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    # Iterate the generator to get image and mask batches
    img, mask = next(gen)
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)
        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img[j])
            else:
                # print(mask.shape)
                for m in range(len(mask[0, 0, 0, :])):
                    mask_one = mask[j, :, :, m]
                    ax.imshow(mask_one, alpha=0.5)
            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()


def rot90_random(img, mask):
    flip_bool = bool(random.getrandbits(1))
    img_list2 = []
    mask_list2 = []
    if flip_bool:
        for indeximg, f in enumerate(img):
            img_list2.append(np.rot90(f))
            # print(indeximg)
            mask_list = []
            for i, jjjjjj in enumerate(mask[0, 0, indeximg]):
                m = mask[indeximg, :, :, i]
                mask_list.append(np.array(np.rot90(m), np.newaxis))
            mask_list = np.array(mask_list).transpose((1, 2, 0))
            mask_list2.append(mask_list)
    else:
        img_list2 = np.array(img)
        mask_list2 = np.array(mask)

    yield np.array(img_list2), np.array(mask_list2)


def edit_background(img, mask_image_all):
    flip_bool = bool(random.getrandbits(1))
    img_list = []
    if flip_bool:
        dir = os.path.join(ROOT_DIR, "rand_back")
        l = os.listdir(dir)
        allfiles = []
        for file in l:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                allfiles.append(os.path.join(dir, file))
        path_img_back = random.choice(allfiles)
        w = img[0].shape[0]
        h = img[0].shape[1]
        background = cv2.imread(os.path.join(ROOT_DIR, path_img_back), flags=cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, dsize=(w, h))

        for index, image in enumerate(img):
            img_n = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            mask_image_12channel = mask_image_all[index]
            full_mask = np.zeros((w, h), dtype="uint8")

            for levelmask in range(len(mask_image_12channel[0, 0, :])):
                mask_one_channel = mask_image_12channel[:, :, levelmask]
                mask_one_channel = np.array(mask_one_channel, dtype=bool)
                np.putmask(full_mask, mask_one_channel, 255)

            mask_bool = np.array(full_mask, dtype=bool)
            background_result = np.zeros_like(img_n)
            background_result[mask_bool] = img_n[mask_bool]
            background_result[~mask_bool] = background[~mask_bool]
            imgss = background_result.astype(np.float64) / 255
            img_list.append(imgss)
    else:
        img_list = img
    np.array(img_list, dtype=np.uint8)
    yield np.array(img_list), np.array(mask_image_all)


def augmentationsGenerator(gen):
    for img, mask in gen:
        gen = rot90_random(img, mask)
        img222, mask222 = next(gen)
        gen2 = edit_background(img222, mask222)
        img222, mask222 = next(gen2)
        img_aug = img222
        mask_aug = mask222
        yield img_aug, mask_aug
