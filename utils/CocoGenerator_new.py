import os.path
import random

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO

from definitions import DATASET_PATH, ROOT_DIR


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

    def on_epoch_end(self):
        self.c = 0
        random.shuffle(self.image_list)

    def __len__(self):
        return int(len(self.image_list) // self.batch_size) * 80

    def getImagePathByCocoId(self, image_id):
        image = self.coco.loadImgs([image_id])[0]
        imagePath = DATASET_PATH + '/' + image['file_name']
        return imagePath

    def getImage(self, imageObj, dir_images):
        imagepath = os.path.join(dir_images, imageObj['file_name'])
        # print(imagepath)
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
        train_mask = np.zeros((self.input_image_size[0], self.input_image_size[1]))
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = self.classes.index(className) + 1
            new_mask = cv2.resize(self.coco.annToMask(
                anns[a]) * pixel_value, self.input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        train_mask = train_mask[:, :, np.newaxis]
        return train_mask.astype(np.float32)

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
        mask_new = mask_new[:, :, np.newaxis]

        return np.array(image), mask_new.astype(np.float32)

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

    def __iter__(self):
        return self.__getitem__(self.c)

    def __getitem__(self, index):
        img = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 3)).astype('float32')
        mask = np.zeros((self.batch_size, self.input_image_size[0], self.input_image_size[1], 1)).astype('float32')
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
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
            # if self.aurgment:
            # train_img, train_mask = self.add_rotate(train_img, train_mask)
            # train_img, train_mask = self.add_noise_blur(train_img, train_mask)
            img[i], mask[i] = train_img, train_mask

        self.c += self.batch_size

        if self.c + self.batch_size >= len(self.image_list):
            self.c = 0
            random.shuffle(self.image_list)
        if self.aurgment:
            img, mask = self.edit_background(img, mask)

        ohe_hot_mask = tf.keras.utils.to_categorical(mask, num_classes=len(self.classes)+1)
        return img, ohe_hot_mask
