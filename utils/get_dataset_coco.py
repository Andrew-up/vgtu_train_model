import os
import random
from pycocotools.coco import COCO

from definitions import DATASET_PATH


def filterDataset(ann_file_path, classes=None, mode='train', percent_valid=50, path_folder=None):

    # initialize COCO api for instance annotations
    annFile = ann_file_path
    # print(annFile)

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

    # Now, filter out the repeated images
    if classes is None:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)

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
                imagePath = os.path.join(DATASET_PATH, path_folder)
                imagePath = os.path.join(imagePath, image_one['file_name'])
            if os.path.exists(imagePath):
                valid_files.append(image_one)
        group_class.append(valid_files)


    # print(group_class)

    # return 0
    images_train_tmp = []
    images_valid_tmp = []


    # image = coco.loadImgs([266])[0]
    # imagePath = DATASET_PATH +'/'+ image['file_name']
    # print(imagePath)
    # file_true = os.path.exists(imagePath)
    # print(file_true)

    images_train_unique = []
    images_valid_unique = []
    for classesss in group_class:
        if percent_valid > 0:
            b = round(percent_valid / 100 * len(classesss))
            images_train_tmp += classesss[b:]
            images_valid_tmp += classesss[:b]
        else:
            images_train_tmp += classesss


    #
    # a = len(group_class[0])
    # b = round(percent / 100 * len(group_class[0]))

    # print(group_class[0][:b])


    # fdfdfd = group_class[0][:b]
    # for i in fdfdfd:
    #     # imgone = i['id']
    #     # print(i)
    #     annIds = coco.getAnnIds(imgIds=i['id'])
    #     anns = coco.loadAnns(annIds)
    #     # print(anns)
    #     cat_ids = coco.getCatIds(catIds=anns[0]['category_id'])
    #     cats = coco.loadCats(cat_ids)
    #     # print(cats[0]['name'])
    #     # print(idddd)
    # # print(coco.cats)

    for i in range(len(images_train_tmp)):
        if images_train_tmp[i] not in images_train_unique:
            images_train_unique.append(images_train_tmp[i])

    for i in range(len(images_valid_tmp)):
        if images_valid_tmp[i] not in images_valid_unique:
            images_valid_unique.append(images_valid_tmp[i])

    print('____________________________')
    # print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ - : {len(images_train_unique)}')
    # print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ - : {len(images_valid_unique)}')



    #
    # for i in range(len(images)):
    #     if images[i] not in unique_images:
    #         unique_images.append(images[i])

    # print(len(images_train_unique))
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


# def getDataset(batch_size, subset, ):
#     dataset = DataGeneratorFromCocoJson(batch_size=8,
#                                         subset='train',
#                                         image_list=images_train,
#                                         classes=classes,
#                                         input_image_size=input_image_size,
#                                         ann_file=path_json_train,
#                                         shuffle=False)
#
#     return dataset
