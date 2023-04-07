import os.path
import random

import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from pycocotools.coco import COCO
from definitions import DATASET_PATH




class DataGeneratorFromCocoJson(tf.keras.utils.Sequence):
    def __init__(self, batch_size=8, subset="train", image_list=[], classes=[], input_image_size=(128, 128),
                 shuffle=False, coco: COCO = None,
                 path_folder=None):

        super().__init__()
        self.subset = subset
        self.batch_size = batch_size
        self.indexes = np.arange(len(image_list))
        self.image_list = image_list
        self.classes = classes
        self.input_image_size = (input_image_size)
        self.dataset_size = len(image_list)
        self.coco = coco
        catIDs = self.coco.getCatIds(catNms=self.classes)
        self.catIds = catIDs
        self.cats = self.coco.loadCats(catIDs)
        self.imgIds = self.coco.getImgIds()
        self.path_folder_image = path_folder
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.image_list) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def getClassName(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id'] == classID:
                return cats[i]['name']
        return None

    def getNormalMask(self, image_id, catIds):
        # print(image_id)
        annIds = self.coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        # print(anns)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros(self.input_image_size, dtype=np.uint8)
        for a in range(len(anns)):

            # print(anns[a])
            # print('=============')
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className) + 1
            # print(self.coco.annToMask(anns[a]))

            new_mask = cv2.resize(self.coco.annToMask(
                anns[a]) * pixel_value, self.input_image_size)

            train_mask = np.maximum(new_mask, train_mask)
            # train_mask = new_mask / 255.0
        # plt.imshow(train_mask)
        # plt.show()
        return train_mask

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
        return res

    def getImage(self, file_path):
        train_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        # train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
        train_img = cv2.resize(train_img, (self.input_image_size))
        train_img = train_img.astype(np.float32) / 255.
        if (len(train_img.shape) == 3 and train_img.shape[2] == 3):
            return train_img
        else:
            stacked_img = np.stack((train_img,) * 3, axis=-1)
            return stacked_img

    def get_image_Infos_by_path_id(self, node):
        for dict in self.image_list:
            if dict['file_name'] == node:
                return dict

    def getImagePathByCocoId(self, image_id):
        image = self.coco.loadImgs([image_id])[0]
        imagePath = DATASET_PATH +'/'+ image['file_name']
        if self.path_folder_image is not None:
            imagePath = f'{DATASET_PATH}/{self.path_folder_image}/{image["file_name"]}'
            # print('folder is not none')
            # print(imagePath)
        # print(imagePath)
        return imagePath

    def flip_random(self, img, mask):
        flip_bool = bool(random.getrandbits(1))
        if flip_bool:
            img = np.flip(img)
            mask_list = []
            for i in mask:
                mask_list.append(np.flip(i))
            return img, mask_list
        return img, mask

    def rot90_random(self, img, mask):
        flip_bool = bool(random.getrandbits(1))
        if flip_bool:
            img = np.rot90(img)
            mask_list = []
            for i in mask:
                mask_list.append(np.rot90(i))
            return img, mask_list
        return img, mask


    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3))
        y = np.empty((self.batch_size, 128, 128, len(self.classes)))
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # print(range(len(indexes)))
        # print('11111111111111111111111111111111')
        # return 0
        for i in range(len(indexes)):
            value = indexes[i]
            img_info = self.image_list[value]
            # w = img_info['height']
            # h = img_info['width']
            # X[i, ] = self.getImage(self.getImagePathByCocoId(img_info['id']))
            img = self.getImage(self.getImagePathByCocoId(img_info['id']))
            # print(X.shape)
            # print('______________________')
            # plt.imshow(self.getImage(getImagePathById(img_info['id'])))
            # print('sssssssssssss')
            mask_train = self.getLevelsMask(img_info['id'])
            X[i, ] = img
            # X[i, ], mask_train = self.flip_random(img, mask_train)
            # X[i, ], mask_train = self.rot90_random(X[i, ], mask_train)
            # X[i, ] = tf.image.random_brightness((X[i, ]*255).astype(np.uint8), 0.2)
            # X[i, ] = tf.image.random_contrast(X[i, ], 0.5, 0.8) / 255




            # plt.imshow(X[i, ])
            # plt.show()

            # print(X[i, ])
            # print(type(mask_train))
            # print(type(X[i, ]))

            # print(len(mask_train))


            # print(i)
            # print(self.catIds)
            for j in self.catIds:

                # print('==============')
                # print(f'i: {i}')
                # print(f'j: {j}')
                y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                #
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                #
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                #
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]

                # y[i, :, :, j-1] = mask_train[j-1]
                # y[i, :, :, j-1] = mask_train[j-1]



        X = np.array(X)
        y = np.array(y)
        # print('111111111111111')

        if self.subset == 'train':
            return X, y
        else:
            return X
