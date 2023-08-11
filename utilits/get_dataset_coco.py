import os
import random
from pycocotools.coco import COCO

from definitions import DATASET_PATH


def filterDataset(ann_file_name, classes=None, mode='train', percent_valid=50, path_folder=None, shuffie=True):

    weight_list = [0.3]
    # initialize COCO api for instance annotations
    annFile = DATASET_PATH + 'annotations/' + ann_file_name
    annFile = os.path.normpath(annFile)
    print(annFile)

    coco = COCO(annFile)
    print("filterDataset")
    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:

            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    if classes is None:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)


    categories = coco.loadCats(coco.getCatIds())
    # перебираем категории
    for category in categories:
        # получаем id категории
        category_id = category['id']
        # получаем аннотации для данной категории
        ann_ids = coco.getAnnIds(catIds=[category_id])
        # если есть аннотации, то добавляем 1 в список, иначе 0
        if len(ann_ids) > 0:
            weight_list.append(1.0)
        else:
            weight_list.append(0.0)

    group_class = []

    for i in classes:
        l = []
        catIds = coco.getCatIds(catNms=i)
        imgssss = coco.getImgIds(catIds=catIds)
        l += coco.loadImgs(imgssss)
        valid_files = []
        for image_one in l:
            imagePath = DATASET_PATH + path_folder + '/' + image_one['file_name']
            imagePath = os.path.normpath(imagePath)
            if os.path.exists(imagePath):
                valid_files.append(image_one)
            else:
                print(f'no image : {imagePath}')
        group_class.append(valid_files)

    images_train_tmp = []
    images_valid_tmp = []


    images_train_unique = []
    images_valid_unique = []
    for classesss in group_class:
        if percent_valid > 0:
            b = round(percent_valid / 100 * len(classesss))
            images_train_tmp += classesss[b:]
            images_valid_tmp += classesss[:b]
        else:
            images_train_tmp += classesss

    for i in range(len(images_train_tmp)):
        if images_train_tmp[i] not in images_train_unique:
            images_train_unique.append(images_train_tmp[i])

    for i in range(len(images_valid_tmp)):
        if images_valid_tmp[i] not in images_valid_unique:
            images_valid_unique.append(images_valid_tmp[i])



    if shuffie:
        random.shuffle(images_train_unique)
        random.shuffle(images_valid_unique)


    if classes is not None:
        return images_train_unique, images_valid_unique, coco, classes, weight_list
    else:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)
        return images_train_unique, images_valid_unique, coco, classes, weight_list
