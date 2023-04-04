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

    def __len__(self):
        return int(len(self.image_list) / self.batch_size)

    def getImagePathByCocoId(self, image_id):
        image = self.coco.loadImgs([image_id])[0]
        imagePath = DATASET_PATH + '/' + image['file_name']
        return imagePath

    def getImage(self, imageObj,  dir_images):
        imagepath = os.path.join(dir_images, imageObj['file_name'])
        if not os.path.exists(imagepath):
            print(f'Не могу найти путь: {imagepath}')
        train_img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
        train_img = (np.array(cv2.resize(train_img, self.input_image_size)) / 255).astype(np.float32)
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
        return train_mask

    def __next__(self):
        return self.__getitem__(self.batch_size)


    def __iter__(self):
        return self

    def __getitem__(self, index):
        img = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 3)).astype('float')
        mask = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 1)).astype('float')
        # print()
        # print(f'c: {self.c}')
        # print(f'batch size: {self.batch_size}')
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        for i in range(len(indexes)):
            if not self.image_list:
                print('СПИСОК ИЗОБРАЖЕНИЙ ПУСТ')
                break
            value = indexes[i]
            img_info = self.image_list[value]
            # imageObj = self.image_list[i]
            train_img = self.getImage(imageObj=img_info, dir_images=self.img_folder)
            train_mask = np.zeros((self.input_image_size[0], self.input_image_size[1], 1))
            if self.mask_type == 'categorical':
                train_mask = self.getNormalMask(img_info['id'])
                pass
            img[i] = train_img
            mask[i] = train_mask

            pass
        self.c += self.batch_size
        # visualizeImageOrGenerator(images_list=np.array(img), mask_list=np.array(mask))
        # time.sleep(1)

        if self.c + self.batch_size >= len(self.image_list):
            # print('перемешиваю список изображений')
            self.c = 0
            random.shuffle(self.image_list)

        return np.array(img).astype(np.float32), np.array(mask).astype(np.uint8)


