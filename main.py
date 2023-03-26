import os.path
import time

import matplotlib.pyplot as plt
import numpy as np

from definitions import MODEL_H5_PATH, ANNOTATION_FILE_PATH, MODEL_H5_FILE_NAME, ANNOTATION_FILE_PATH_TEST, \
    ANNOTATION_FILE_PATH_TRAIN, ANNOTATION_FILE_PATH_VALID
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.build_model import unet_model
from utils.get_dataset_coco import filterDataset
from utils.model_train import train_model
from utils.vizualizators import vizualizator, show_mask_true_and_predict
from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history, get_category_not_null
from datetime import datetime
import zipfile
from utils.helpers import delete_legacy_models_and_zip
from utils.model_losses import plot_segm_history

from utils.newDataGeneratorCoco import ssssssssssssssssssssssssaaaaaaaaaaaaaaaaaa, hui, visualizeGenerator, augmentationsGenerator

from utils.unet import get_model


def main():
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=2)
    if check_garbage_files_count == 0:
        print('Мусора нет')
    else:
        print(f'Удалено старых моделей h5 и zip архивов: {check_garbage_files_count}')
    timer = time.time()

    images_train, _, coco_train, classes_train = filterDataset(ANNOTATION_FILE_PATH,
                                                               percent_valid=0,
                                                               # path_folder='train'
                                                               )
    images_valid, _, coco_valid, classes_valid = filterDataset(ANNOTATION_FILE_PATH,
                                                               percent_valid=0,
                                                               # path_folder='valid'
                                                               )

    # print('classes_train: ')
    # print(classes_train)
    # images_test, _, coco, classes = filterDataset(ANNOTATION_FILE_PATH_TEST, percent_valid=0)

    # print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ - : {len(images_train)}')
    # print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ - : {len(images_valid)}')

    h, w, n_c = 128, 128, len(classes_train)

    # for i in range(10):
    batch_size = 7

    sssssssss11111111 = ssssssssssssssssssssssssaaaaaaaaaaaaaaaaaa(images_train,
                                                                   classes_train,
                                                                   coco_train,
                                                                   folder='train',
                                                                   mask_type="normal",
                                                                   input_image_size=(128, 128),
                                                                   batch_size=batch_size)


    aug_gen = augmentationsGenerator(sssssssss11111111)
    aug_gen_val = augmentationsGenerator(sssssssss11111111)


    # train_generator = DataGeneratorFromCocoJson(batch_size=batch_size,
    #                                             subset='train',
    #                                             # path_folder='train',
    #                                             input_image_size=(128, 128),
    #                                             image_list=images_train,
    #                                             classes=classes_train,
    #                                             coco=coco_train,
    #                                             shuffle=True)
    #
    # valid_generator = DataGeneratorFromCocoJson(batch_size=batch_size,
    #                                             # path_folder='valid',
    #                                             subset='train',
    #                                             input_image_size=(128, 128),
    #                                             image_list=images_valid,
    #                                             classes=classes_valid,
    #                                             coco=coco_valid,
    #                                             shuffle=True)


    # visualizeGenerator(sssssssss11111111)
    visualizeGenerator(aug_gen)
    visualizeGenerator(aug_gen_val)
    # return 0
    # return 0

    # aug_gen = augmentationsGenerator(val_gen, augGeneratorArgs)

    print(f'h, w, n_c: {h, w, n_c}')
    # i, m = train_generator.__getitem__(0)
    # vizualizator(i, m, classes_train)

    # return 0
    iou_valid = get_category_not_null()
    model = get_model(img_size=(128, 128, 3),
                      num_classes=len(classes_train),
                      iou_valid=iou_valid)

    path_model = os.path.join(MODEL_H5_PATH, MODEL_H5_FILE_NAME)
    model_history = get_last_model_history()
    if model_history:
        path_model = os.path.join(MODEL_H5_PATH, model_history.name_file)
    print(path_model)
    history = train_model(path_model=path_model,
                          model=model,
                          n_epoch=50,
                          batch_size=batch_size,
                          dataset_train=aug_gen,
                          dataset_valid=aug_gen_val,
                          dataset_size_train=len(images_train),
                          model_history=model_history,
                          monitor='my_mean_iou')

    plot_segm_history(history, metrics=['my_mean_iou', 'val_my_mean_iou'])

    # show_mask_true_and_predict()

    path_zip = zipfile.ZipFile(f'{os.path.splitext(path_model)[0]}.zip', 'w')
    path_zip.write(path_model, arcname=f'{model_history.name_file}')
    path_zip.close()

    model.save(path_model)

    if model_history:
        model_history.date_train = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        model_history.quality_dataset = len(images_train) + len(images_valid)
        model_history.quality_valid_dataset = str(len(images_valid))
        model_history.quality_train_dataset = str(len(images_train))
        model_history.num_classes = str(len(classes_train))
        model_history.status = "compleated"
        model_history.input_size = str(model.input_shape)
        model_history.output_size = str(model.output_shape)
        model_history.time_train = time.strftime("время обучения: %H часов %M минут %S секунд",
                                                 time.gmtime(time.time() - timer))
        update_model_history(model_history)


if __name__ == "__main__":
    main()
    # viz_model()
