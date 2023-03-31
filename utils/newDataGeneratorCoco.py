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
from definitions import DATASET_PATH, ROOT_DIR, ANNOTATION_FILE_PATH_VALID_IMAGE


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
    null = np.zeros((128, 128), dtype=np.uint8)
    res.append(null)
    for j, categorie in enumerate(catIds):
        mask = getNormalMask(coco=coco, image_id=image_id, catIds=categorie, input_image_size=input_image_size,
                             classes=catIds)
        res.append(mask)
    return res


def getImage(file_path, input_image_size):
    train_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # plt.imshow(train_img)
    # plt.show()
    # train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    train_img = cv2.resize(train_img, (input_image_size))
    train_img = train_img.astype(np.float32) / 255.
    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):
        return train_img
    else:
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getImagePathByCocoId(coco, image_id, folder=None):
    image = coco.loadImgs([image_id])[0]
    imagePath = DATASET_PATH + '/' + image['file_name']
    if folder is not None:
        pass
        imagePath = ANNOTATION_FILE_PATH_VALID_IMAGE + image['file_name']

    # print('folder is not none')
    # print(imagePath)
    # print(imagePath)
    return imagePath


def merge_masks(mask):
    mask = mask
    # Инициализируем новую маску с одним каналом
    merged_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    # Присваиваем каждому пикселю в новой маске значение класса в зависимости от максимального значения в каждом канале
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            channel_values = mask[i, j, :]
            class_id = np.argmax(channel_values)
            merged_mask[i, j, 0] = class_id

    return merged_mask


colors_map = [
    [255, 0, 0],  # класс 1 - красный цвет
    [0, 255, 0],  # класс 2 - зеленый цвет
    [0, 0, 255],  # класс 3 - синий цвет
    [255, 255, 0],  # класс 4 - желтый цвет
    [255, 0, 255],  # класс 5 - фиолетовый цвет
    [0, 255, 255],  # класс 6 - голубой цвет
    [128, 0, 0],  # класс 7 - темно-красный цвет
    [0, 128, 0],  # класс 8 - темно-зеленый цвет
    [0, 0, 128],  # класс 9 - темно-синий цвет
    [128, 128, 0],  # класс 10 - темно-желтый цвет
    [128, 0, 128],  # класс 11 - темно-фиолетовый цвет
    [255, 125, 255],  # новый цвет - белый цвет
]

def visualize_mask(mask):

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            class_id = int(mask[i, j, 0])
            if class_id in colors_map:
                color_mask[i, j, :] = colors_map[class_id]

    # Отображаем цветовую маску
    plt.imshow(color_mask)
    plt.axis('off')
    plt.show()

def semantic_segmentation(mask):
    # получаем индекс канала с максимальным значением в каждом пикселе
    mask_1_channel = np.argmax(mask, axis=-1, keepdims=True)
    return mask_1_channel

def colorize_mask(mask):
    # задаем 15 цветовых классов (0-15)

    colors = [
        [0, 0, 0],       # класс 0 - черный цвет
        [255, 0, 0],     # класс 1 - красный цвет
        [0, 255, 0],     # класс 2 - зеленый цвет
        [0, 0, 255],     # класс 3 - синий цвет
        [255, 255, 0],   # класс 4 - желтый цвет
        [255, 0, 255],   # класс 5 - фиолетовый цвет
        [0, 255, 255],   # класс 6 - голубой цвет
        [128, 0, 0],     # класс 7 - темно-красный цвет
        [0, 128, 0],     # класс 8 - темно-зеленый цвет
        [0, 0, 128],     # класс 9 - темно-синий цвет
        [128, 128, 0],   # класс 10 - темно-желтый цвет
        [128, 0, 128],   # класс 11 - темно-фиолетовый цвет
        [255, 255, 255],  # класс 12 - белый цвет
    ]

    image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            class_id = int(mask[i, j, 0])
            if class_id in colors_map:
                image[i, j, :] = colors_map[class_id]

    return image

def color_mask(mask):
    # Создаем пустой массив для раскрашенной маски
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Создаем словарь с цветами для каждого класса
    colors_dict = {
        0: [0, 0, 0],  # Черный
        1: [0, 255, 0],  # Зеленый
        2: [0, 0, 255],  # Синий
        3: [255, 255, 0],  # Желтый
        4: [0, 255, 255],  # Бирюзовый
        5: [128, 0, 128],  # Фиолетовый
        6: [255, 165, 0],  # Оранжевый
        7: [255, 192, 203],  # Розовый
        8: [128, 128, 128],  # Серый
        9: [165, 42, 42],  # Коричневый
        10: [128, 128, 0],  # Оливковый
        11: [0, 255, 0],  # Лайм
        12: [0, 128, 128],  # Морской волны
        13: [255, 255, 255],  # Белый
        14: [255, 0, 0]  # Красный

    }
    # Проходимся по всем пикселям маски
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Получаем класс текущего пикселя
            cls = mask[i, j, 0]
            # Получаем цвет для данного класса из словаря
            color = colors_dict[cls]
            # Раскрашиваем пиксель в соответствующий цвет
            colored_mask[i, j, :] = color
    return colored_mask

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
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')
        # print(img.shape)
        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            file_path_image = getImagePathByCocoId(coco, image_id=imageObj["id"], folder=folder)
            train_img = getImage(file_path=file_path_image, input_image_size=input_image_size, )
            mask_train = getLevelsMask(coco, imageObj['id'], catIds, input_image_size)
            mask_train_numpy = np.array(mask_train).transpose((1, 2, 0))
            mask_1_channel = semantic_segmentation(mask_train_numpy).astype(np.uint8)

            plt.imshow(color_mask(mask_1_channel))
            mask[i - c] = mask_1_channel
            img[i - c] = train_img

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


def gasuss_noise(image, koef):
    uni_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.randu(uni_noise, 0, 255)
    uni_noise = (uni_noise * koef).astype(np.uint8)
    uni_merge = cv2.merge((uni_noise, uni_noise, uni_noise))
    image = (image * 255).astype(np.uint8)
    gn_img = cv2.add(image, uni_merge)

    return np.array(gn_img / 255).astype(np.float64)


def add_rotate(img_batch, mask_batch, angle=45, scale=1.0):
    new_img_list = []
    new_mask_list = []
    for indeximg, f in enumerate(img_batch):
        angle = random.randrange(0, 359)
        # print(f'angle: {angle} ')
        (h, w) = f.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(f, M, (w, h))
        new_img_list.append(rotated)
        mmmmm = []
        for i in range(len(mask_batch[0, 0, 0, :])):
            mask = np.zeros((w, h), dtype=np.float64)
            new_mask = cv2.warpAffine(mask_batch[indeximg, :, :, i], M, mask.shape)
            mmmmm.append(new_mask)
        mmmmm = np.array(mmmmm).transpose((1, 2, 0))
        new_mask_list.append(mmmmm)
    new_mask_list = np.array(new_mask_list)
    # print(new_mask_list)
    return np.array(new_img_list), new_mask_list


def add_noise_blur(img_batch, mask_batch):
    img_list2 = []

    for indeximg, f in enumerate(img_batch):
        random_koef = random.uniform(0, 0.3)
        f = gasuss_noise(f, random_koef)
        img_list2.append(f)

    img_list2 = np.array(img_list2)
    mask_list2 = np.array(mask_batch)

    return np.array(img_list2), np.array(mask_list2)


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
        background = cv2.resize(background, dsize=(w, h), interpolation=cv2.INTER_AREA)

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


def augmentationsGenerator(gen, mode: str = 'none'):
    for img, mask in gen:
        # img222, mask222 = add_rotate(img_batch=img, mask_batch=mask)
        # img222, mask222 = next(gen)
        #    gen2 = edit_background(img222, mask222)
        #   img222, mask222 = next(gen2)
        img_aug = img.astype(np.float64)
        mask_aug = mask.astype(np.float64)
        yield img_aug, mask_aug