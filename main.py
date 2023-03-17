import os.path
import time

import matplotlib.pyplot as plt

from definitions import MODEL_H5_PATH, ANNOTATION_FILE_PATH, MODEL_H5_FILE_NAME
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.build_model import unet_model
from utils.get_dataset_coco import filterDataset
from utils.model_train import train_model
from utils.vizualizators import vizualizator
from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history
from datetime import datetime
import zipfile
from utils.helpers import delete_legacy_models_and_zip


def main():
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=2)
    if check_garbage_files_count == 0:
        print('Мусора нет')
    else:
        print(f'Удалено старых моделей h5 и zip архивов: {check_garbage_files_count}')
    timer = time.time()
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=30)

    h, w, n_c = 128, 128, len(classes)

    train_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)

    valid_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_valid,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)

    img_list, img_mask = train_generator_class.__getitem__(1)
    vizualizator(img_list, img_mask)

    model = unet_model(len(classes), (h, w, n_c))
    path_model = os.path.join(MODEL_H5_PATH, MODEL_H5_FILE_NAME)
    model_history = get_last_model_history()
    if model_history:
        path_model = os.path.join(MODEL_H5_PATH, model_history.name_file)

    history = train_model(path_model=path_model,
                          model=model,
                          n_epoch=1,
                          batch_size=8,
                          dataset_train=train_generator_class,
                          dataset_valid=valid_generator_class,
                          dataset_size_train=len(images_train),
                          model_history=model_history)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for IOU
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('dice_coef iou')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    path_zip = zipfile.ZipFile(f'{os.path.splitext(path_model)[0]}.zip', 'w')
    path_zip.write(path_model, arcname=f'{model_history.name_file}')
    path_zip.close()

    model.save(path_model)

    if model_history:
        model_history.date_train = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        model_history.quality_dataset = len(images_train) + len(images_valid)
        model_history.quality_valid_dataset = str(len(images_valid))
        model_history.quality_train_dataset = str(len(images_train))
        model_history.num_classes = str(len(classes))
        model_history.status = "compleated"
        model_history.input_size = str(model.input_shape)
        model_history.output_size = str(model.output_shape)
        model_history.time_train = time.strftime("время обучения: %H часов %M минут %S секунд",
                                                 time.gmtime(time.time() - timer))
        update_model_history(model_history)


if __name__ == "__main__":
    main()
    # viz_model()
