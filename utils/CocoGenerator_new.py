import os.path
import random
import time

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
from utils.newDataGeneratorCoco import visualizeImageOrGenerator
from utils.vizualizators import vizualizator_old


class NEWJSON_COCO_GENERATOR(tf.keras.utils.Sequence):
    def __init__(self, batch_size=8, image_list=[], classes=[], input_image_size=(128, 128),
                 shuffle=False, coco: COCO = None,
                 path_folder=None,
                 mask_type='categorical'):

        super().__init__()
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

    def on_epoch_end(self):
        self.c = 0
        random.shuffle(self.image_list)

    def __len__(self):
        return 10000
        # return int(len(self.image_list)//self.batch_size)

    def getImagePathByCocoId(self, image_id):
        image = self.coco.loadImgs([image_id])[0]
        imagePath = DATASET_PATH + '/' + image['file_name']
        return imagePath

    def getImage(self, imageObj,  dir_images):
        imagepath = os.path.join(dir_images, imageObj['file_name'])
        # print(imagepath)
        if not os.path.exists(imagepath):
            print(f'Не могу найти путь: {imagepath}')
        train_img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
        train_img = (np.array(cv2.resize(train_img, self.input_image_size))/255).astype(np.float32)
        if len(train_img.shape) == 3 and train_img.shape[2] == 3:
            return train_img
        else:
            stacked_img = np.stack((train_img,) * 3, axis=-1)
            return stacked_img

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return None

    def gasuss_noise(self, image, koef):
        uni_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.randu(uni_noise, 0, 255)
        uni_noise = (uni_noise * koef).astype(np.uint8)
        uni_merge = cv2.merge((uni_noise, uni_noise, uni_noise))
        image = (image * 255).astype(np.uint8)
        gn_img = cv2.add(image, uni_merge)
        return np.array(gn_img / 255).astype(np.float64)
    def add_noise_blur(self, image, mask):
        random_koef = random.uniform(0, 0.3)
        image = self.gasuss_noise(image, random_koef)
        return np.array(image), np.array(mask)

    def add_rotate(self, image, mask, angle=45, scale=1.0):
        angle = random.randrange(0, 359)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, M, (w, h))
        mask_new = cv2.warpAffine(mask, M, (w, h)).astype(np.uint8)
        # mask_new = mask[:, :, np.newaxis]

        return np.array(image), mask.astype(np.float32)

    def getNormalMask(self, image_id):
        annIds = self.coco.getAnnIds(image_id, catIds=self.catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(self.catIds)
        train_mask = np.zeros(self.input_image_size)
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className) + 1
            new_mask = cv2.resize(self.coco.annToMask(
                anns[a]) * pixel_value, self.input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        # print('Unique pixel values in the mask are:', np.unique(train_mask))
        train_mask = train_mask[:, :, np.newaxis]
        return train_mask.astype(np.float32)

    def __next__(self):
        print('next')
        return self.__getitem__(self.c)


    def __getitem__(self, index):

        # print(f'----------- c: {self.c}')
        img = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 3)).astype('float32')
        mask = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 1)).astype('float32')
        indexes = self.indexes[index*self.batch_size: (index + 1) * self.batch_size]
        # print(f'indexes: {indexes}')
        # print(f'index: {index}')
        # print(f'c: {self.c}')

        train_aug = ImageDataGenerator(
            # rescale=1. / 255,
            horizontal_flip=True,
            vertical_flip=True
            # height_shift_range=0.9,
            # width_shift_range=0.9,
            # brightness_range=(0.9, 11.5),
            # zoom_range=[5, 11.5],
        )
        for i in range(len(indexes)):

            if not self.image_list:
                print('СПИСОК ИЗОБРАЖЕНИЙ ПУСТ')
                break
            value = indexes[i]

            img_info = self.image_list[value]
            train_img = self.getImage(imageObj=img_info, dir_images=self.img_folder)
            train_mask = np.zeros((self.input_image_size[0], self.input_image_size[1], 1))
            if self.mask_type == 'categorical':
                train_mask = self.getNormalMask(img_info['id'])
            # train_img, train_mask = self.add_rotate(train_img, train_mask)
            # train_img, train_mask = self.add_noise_blur(train_img, train_mask)



            img[i] = train_img
            mask[i] = train_mask
        self.c += self.batch_size

        if self.c + self.batch_size >= len(self.image_list):
            # print('перемешиваю список')
            self.c = 0
            random.shuffle(self.image_list)

        g_x = train_aug.flow(img, batch_size=img.shape[0], shuffle=True)

        plt.imshow(g_x.x[0])
        plt.show()

        plt.imshow(img[0])
        plt.show()

        # return np.array(xxxxx.x), np.array(xxxxx.y)


