import math
import os.path
import random
from typing import Union, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pycocotools.coco import COCO
import keras.backend as K
from definitions import DATASET_PATH, ROOT_DIR

colors = [
    [0, 0, 0],  # Красный
    [0, 255, 0],  # Зеленый
    [0, 0, 255],  # Синий
    [255, 255, 0]  # Желтый
]


def color_mask(mask):
    mask = np.argmax(mask, axis=-1)
    mask = mask[:, :, np.newaxis]
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Получаем класс текущего пикселя
            cls = mask[i, j, 0]
            # Получаем цвет для данного класса из словаря
            color = colors[cls]
            # Раскрашиваем пиксель в соответствующий цвет
            colored_mask[i, j, :] = color
    return colored_mask


class DatasetGeneratorFromCocoJson(tf.keras.utils.Sequence):
    def __init__(self, batch_size=8, image_list=[], classes=[], input_image_size=(128, 128),
                 shuffle=False, coco: COCO = None,
                 path_folder=None,
                 mask_type='categorical',
                 aurgment=True):

        super().__init__()
        self.aurgment = aurgment
        self.batch_size = batch_size
        self.image_list = image_list
        self.classes = classes
        self.indexes = np.arange(len(image_list))
        self.input_image_size = (input_image_size)
        self.dataset_size = len(image_list)
        self.coco = coco
        self.test = 0
        self.c = 0
        self.catIds = self.coco.getCatIds(catNms=self.classes)
        self.img_folder = path_folder
        self.mask_type = mask_type
        self.index = 0

    def __next__(self):
        return self.__getitem__(self.c)
    def getLevelsMask(self, image_id):
        # for each category , we get the x mask and add it to mask list
        res = []
        # mask = np.zeros((self.input_image_size))
        for j, categorie in enumerate(self.catIds):
            # annIds = self.coco.getAnnIds(image_id, catIds=categorie, iscrowd=None)
            # anns = self.coco.loadAnns(annIds)
            # print(image_id)
            # print('SSSSSSSSSSSS')
            mask = self.getNormalMask(image_id, categorie)
            # print(type(mask))
            res.append(mask)
        res = np.array(res)
        # res = res.transpose((1, 2, 0))
        # new_res = np.argmax(res, axis=-1)
        return res

    def on_epoch_end(self):
        self.c = 0
        np.random.shuffle(self.image_list)

    def __len__(self):
        return 999999999
        return math.floor(len(self.image_list) / self.batch_size)

    def getImagePathByCocoId(self, image_id):
        image = self.coco.loadImgs([image_id])[0]
        imagePath = DATASET_PATH + '/' + image['file_name']
        return imagePath

    def getImage(self, imageObj, dir_images):
        imagepath = os.path.join(dir_images, imageObj['file_name'])
        # print(imagepath)
        if not os.path.exists(imagepath):
            print(f'Не могу найти путь: {imagepath}')
        train_img = cv2.imread(imagepath, cv2.COLOR_BGR2RGB)
        #
        # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
        #
        # lab = cv2.cvtColor(train_img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        # l, a, b = cv2.split(lab)  # split on 3 different channels
        #
        # l2 = clahe.apply(l)  # apply CLAHE to the L-channel
        #
        # lab = cv2.merge((l2, a, b))  # merge channels
        # img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        # b, g, r = cv2.split(train_img)
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # im = cv2.filter2D(train_img, -1, kernel)
        # r[:] = r[:]
        # b[:] = 50
        # g[:] = 50
        # print(g)

        # imgMerged = cv2.merge((r, g, b))  # в этот раз соберём сразу RGB
        train_img = cv2.resize(train_img, (self.input_image_size))
        # train_img = cv2.resize(train_img, self.input_image_size) / 255).astype(np.float32)
        train_img = train_img.astype(np.float32) / 255.
        if len(train_img.shape) == 3 and train_img.shape[2] == 3:
            return train_img
        else:
            # print(train_img.shape)
            stacked_img = np.stack((train_img,) * 3, axis=-1)
            # print('stacked_img')
            return stacked_img

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return None


    def checkMaskPixelIsClassId(self, mask, classId):
        newmask = mask.copy()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                # Получаем класс текущего пикселя
                cls = mask[i, j]
                if cls != classId and cls != 0:
                    # print(classId)
                    newmask[i, j] = classId

        return newmask

    def getNormalMask(self, image_id, catIds):
        annIds = self.coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros((self.input_image_size[0], self.input_image_size[1]))
        for a in range(len(anns)):
            # train_mask1111 = np.zeros((512, 512))
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className) + 1
            new_mask = cv2.resize(self.coco.annToMask(
                anns[a]) * pixel_value, self.input_image_size)
            # new_mask = self.checkMaskPixelIsClassId(mask=new_mask, classId=pixel_value)
            train_mask = np.maximum(new_mask, train_mask)
            pass
        train_mask = train_mask[:, :, np.newaxis]
        # train_mask = train_mask.reshape((self.input_image_size[0], self.input_image_size[1], 1))
        return train_mask

    def gasuss_noise(self, image, koef):
        uni_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.randu(uni_noise, 0, 255)
        uni_noise = (uni_noise * koef).astype(np.uint8)
        uni_merge = cv2.merge((uni_noise, uni_noise, uni_noise))
        image = (image * 255).astype(np.uint8)
        gn_img = cv2.add(image, uni_merge)
        return np.array(gn_img / 255).astype(np.float64)

    def add_noise_blur(self, image, mask):
        random_koef = random.uniform(0, 0.2)
        image = self.gasuss_noise(image, random_koef)
        return np.array(image), np.array(mask)

    def add_rotate(self, image, mask, angle=45, scale=1.0):
        angle_list = [0, 90, 180, 270]
        angle = random.choice(angle_list)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (w, h))
        new_list_mask = []
        for i in range(mask.shape[-1]):
            mask_new = cv2.warpAffine(mask[:, :, i], M, (w, h)).astype(np.uint8)
            new_list_mask.append(mask_new)
        new_list_mask = np.array(new_list_mask)
        new_list_mask = np.transpose(new_list_mask, (1, 2, 0))
        return np.array(image), new_list_mask

    def edit_background(self, img, mask_image_all):
        flip_bool = bool(random.getrandbits(1))
        img_list = []
        flip_bool = True
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
            background = cv2.resize(background, dsize=(w, h), interpolation=cv2.INTER_AREA)

            for index, image in enumerate(img):
                img_n = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
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
        np.array(img_list)
        return np.array(img_list), np.array(mask_image_all)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.input_image_size[0], self.input_image_size[1], 3))
        y = np.empty((self.batch_size, self.input_image_size[0], self.input_image_size[1], 1))
        # print(f'i: {i}')



        if (self.c+1) * self.batch_size >= len(self.image_list):
            self.c = 0
            random.shuffle(self.image_list)

        indexes = self.indexes[self.c * self.batch_size: (self.c + 1) * self.batch_size]

        for i in range(len(indexes)):

            if not self.image_list:
                print('СПИСОК ИЗОБРАЖЕНИЙ ПУСТ')
                break

            value = indexes[i]
            img_info = self.image_list[value]
            train_img = self.getImage(imageObj=img_info, dir_images=self.img_folder)

            train_mask = self.getNormalMask(img_info['id'], self.catIds)
            train_mask = np.array(train_mask).astype(np.uint8)
            if self.aurgment:
                train_img, train_mask = self.add_rotate(train_img, train_mask)
                train_img, train_mask = self.add_noise_blur(train_img, train_mask)
            X[i, ] = train_img
            y[i, :, :, :] = train_mask


            # print('Перемешиваю список')

        # if self.aurgment:
        #     X, y = self.edit_background(X, y)

        self.c += 1

        img = np.array(X).astype(np.float32)
        img[img == 0.] = 1
        #     print('null')
        mask = np.array(y).astype(np.float32)
        mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=len(self.classes) + 1)
        mask_one_hot = mask_one_hot[:, :, :, 1:]

        return img, mask_one_hot
