import os
import random
from pycocotools.coco import COCO
from definitions import ANNOTATION_FILE_PATH_VALID_IMAGE

from definitions import DATASET_PATH


def filterDataset(ann_file_path, classes=None, mode='train', percent_valid=50, path_folder=None, shuffie=True):

    # initialize COCO api for instance annotations
    annFile = ann_file_path
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

    # if classes is None:
    #     classes = list()
    #     for i in coco.cats:
    #         name = coco.cats[i]['name']
    #         classes.append(name)
    #         # print(name)

    classes2 = set()
    img_ids = coco.getImgIds()
    classes = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        img_classes = [ann['category_id'] for ann in anns]
        classes2.update(img_classes)

    for class_id in classes2:
        class_name = coco.loadCats(class_id)[0]['name']
        classes.append(class_name)
        print(f'{class_id}: {class_name}')


    unique_images = []

    group_class = []
    for i in classes:
        l = []
        catIds = coco.getCatIds(catNms=i)
        imgssss = coco.getImgIds(catIds=catIds)
        l += coco.loadImgs(imgssss)
        valid_files = []
        for image_one in l:
            imagePath = DATASET_PATH + '/' + image_one['file_name']
            if path_folder is not None:
                # imagePath = os.path.join(DATASET_PATH, path_folder)
                imagePath = os.path.join(ANNOTATION_FILE_PATH_VALID_IMAGE, image_one['file_name'])
            if os.path.exists(imagePath):
                valid_files.append(image_one)
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
        # print(len(images_train_unique))
        random.shuffle(images_valid_unique)

    if classes is not None:
        return images_train_unique, images_valid_unique, coco, classes
    else:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)
        return images_train_unique, images_valid_unique, coco, classes
