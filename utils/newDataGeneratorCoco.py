import os.path
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from definitions import DATASET_PATH, ROOT_DIR, ANNOTATION_FILE_PATH_VALID_IMAGE
import tensorflow as tf

def getNormalMask(coco, image_id, catIds, input_image_size, classes):
    annIds = coco.getAnnIds(image_id, catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], catIds)
        print(className)
        pixel_value = classes.index()
        new_mask = cv2.resize(coco.annToMask(
            anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)
    return train_mask


def getLevelsMask(coco, image_id, catIds, input_image_size):
    res = []
    for j, categorie in enumerate(catIds):
        mask = getNormalMask(coco=coco, image_id=image_id, catIds=categorie, input_image_size=input_image_size,
                             classes=catIds)
        res.append(mask)

    return np.stack(np.array(res), axis=-1)


def getImage(file_path, input_image_size):
    train_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    train_img = cv2.resize(train_img, (input_image_size))
    train_img = train_img.astype(np.float32) / 255.
    # plt.imshow(train_img)
    # plt.show()
    if (len(train_img.shape) == 3 and train_img.shape[2] == 3):
        return train_img
    else:
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getImagePathByCocoId(coco, image_id, folder=None):
    image = coco.loadImgs([image_id])[0]
    imagePath = DATASET_PATH + '/' + image['file_name']
    if folder is not None:
        imagePath = ANNOTATION_FILE_PATH_VALID_IMAGE + image['file_name']

    return imagePath


def cocoDataGenerator(images, classes, coco, folder=None,
                      input_image_size=(224, 224), batch_size=2, mode='train',
                      mask_type='categorical',
                      shuffle=False):
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    c = 0
    if shuffle:
        random.shuffle(images)
    while (True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], len(classes))).astype('float')

        listimg = []
        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            listimg.append(imageObj)
            file_path_image = getImagePathByCocoId(coco, image_id=imageObj["id"], folder=folder)
            train_img = getImage(file_path=file_path_image, input_image_size=input_image_size, )
            mask_train = getNormalMask(coco, imageObj['id'], catIds, input_image_size, classes=classes)

            # print(mask_train.shape)
            mask[i - c, :, :, :] = mask_train
            img[i - c] = train_img

        c += batch_size
        if (c + batch_size >= dataset_size):
            c = 0
            random.shuffle(images)

        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        yield img, mask


def visualizeImageOrGenerator(gen=None, subtitle=None, images_list=None, mask_list=None):
    colors = ['#0044ff', '#ff00fb', '#ff0000', '#2bff00', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    # Iterate the generator to get image and mask batches
    if gen is not None:
        img, mask = next(gen)

        # mask = np.argmax(mask)
    else:
        img, mask = images_list, mask_list
        mask = np.argmax(mask, axis=-1)
        mask = mask[:, :, :, np.newaxis]
        print(mask.shape)
    print()
    print(f'-{subtitle}-LEN IMG:  {img.shape}')
    print(f'-{subtitle}-LEN MASK:  {mask.shape}')
    print()
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(subtitle, fontsize=50, fontweight='bold')
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)
        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img[j])
            else:
                mask_one = mask[j, :, :]
                ax.imshow(mask_one, alpha=1)
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
    return np.array(img_list), np.array(mask_image_all)


def augmentationsGenerator(gen, mode: str = None):
    for img, mask in gen:
        img222, mask222 = add_rotate(img_batch=img.copy(), mask_batch=mask.copy())
        img222, mask222 = add_noise_blur(img222, mask222)
        # img222, mask222 = edit_background(img222, mask222)
        img_aug = img222.astype(np.float32)
        mask_aug = mask222.astype(np.float32)

        yield img_aug, mask_aug
