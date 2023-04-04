import os.path
import time
import zipfile
from datetime import datetime

from keras_preprocessing.image import ImageDataGenerator

from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history
from definitions import MODEL_H5_PATH, ANNOTATION_FILE_PATH, MODEL_H5_FILE_NAME, DATASET_PATH
from utils.build_model import unet_model
from utils.get_dataset_coco import filterDataset
from utils.helpers import delete_legacy_models_and_zip
from utils.model_losses import plot_segm_history
from utils.model_train import train_model
from utils.newDataGeneratorCoco import cocoDataGenerator, augmentationsGenerator, visualizeImageOrGenerator
from utils.unet import get_model, unet123123ttt
import tensorflow as tf
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson

from utils.CocoGenerator_new import NEWJSON_COCO_GENERATOR

from utils.vizualizators import vizualizator_old


def main():
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=10)
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
                                                               # path_folder='train'
                                                               )

    train_gen = NEWJSON_COCO_GENERATOR(batch_size=8, image_list=images_train, coco=coco_train, path_folder=DATASET_PATH, classes=classes_train)
    train_val = NEWJSON_COCO_GENERATOR(batch_size=8, image_list=images_train, coco=coco_train, path_folder=DATASET_PATH, classes=classes_train)

    # print(next(train_gen))
    # return 0
    # for i in range(5):
    #     visualizeImageOrGenerator(gen=train_gen)
    # visualizeImageOrGenerator(gen=train_gen)
    # return 0

    # images_valid, _, coco_valid, classes_valid = filterDataset(ANNOTATION_FILE_PATH_VALID,
    #                                                            percent_valid=0,
    #                                                            path_folder='valid'
    #                                                            )

    # print('classes_train: ')
    # print(classes_train)
    # images_test, _, coco, classes = filterDataset(ANNOTATION_FILE_PATH_TEST, percent_valid=0)

    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ - : {len(images_train)}')
    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ - : {len(images_valid)}')

    input_image_size = (128, 128)
    batch_size = 4

    # visualizeImageOrGenerator(gen=train_gen)
    # vizualizator_old(train_gen, classes_train)

    # Вроде норм обучается
    model = get_model((input_image_size[0], input_image_size[1], 3), num_classes=len(classes_train))
    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    path_model = os.path.join(MODEL_H5_PATH, MODEL_H5_FILE_NAME)
    model_history = get_last_model_history()
    if model_history:
        path_model = os.path.join(MODEL_H5_PATH, model_history.name_file)
    print(path_model)

    history = train_model(path_model=path_model,
                          model=model,
                          n_epoch=500,
                          batch_size=batch_size,
                          dataset_train=train_gen,
                          dataset_valid=train_val,
                          dataset_size_train=len(images_train),
                          dataset_size_val=len(images_valid),
                          model_history=model_history,
                          monitor='my_mean_iou',
                          mode='max'
                          )

    plot_segm_history(history, metrics=['my_mean_iou', 'val_my_mean_iou'])

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
        model_history.status = "completed"
        model_history.input_size = str(model.input_shape)
        model_history.output_size = str(model.output_shape)
        model_history.time_train = time.strftime("время обучения: %H часов %M минут %S секунд",
                                                 time.gmtime(time.time() - timer))
        update_model_history(model_history)


if __name__ == "__main__":
    main()
    # viz_model()
