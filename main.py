import os.path
import time
import zipfile
from datetime import datetime

import tensorflow as tf

from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history
from definitions import MODEL_H5_PATH, MODEL_H5_FILE_NAME, DATASET_PATH
from utils.CocoGenerator_new import DatasetGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.helpers import delete_legacy_models_and_zip
from utils.model_losses import plot_segm_history
from utils.model_train import train_model
from utils.unet import unet
from utils.vizualizators import gen_viz


def main():
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=10)
    if check_garbage_files_count == 0:
        print('Мусора нет')
    else:
        print(f'Удалено старых моделей h5 и zip архивов: {check_garbage_files_count}')
    timer = time.time()

    images_train, _, coco_train, classes_train = filterDataset(ann_file_name='_annotations.coco.json',
                                                               percent_valid=0,
                                                               path_folder='train'
                                                               )

    # return 0
    images_valid, _, coco_valid, classes_valid = filterDataset(ann_file_name='_annotations.coco.json',
                                                               percent_valid=0,
                                                               path_folder='valid'
                                                               )

    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ТРЕНИРОВКИ: {len(images_train)}')
    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ: {len(images_valid)}')

    print(classes_train)

    batch_size = 4
    train_gen = DatasetGeneratorFromCocoJson(batch_size=batch_size, image_list=images_train, coco=coco_train,
                                             path_folder=os.path.join(DATASET_PATH, 'train'), classes=classes_train,
                                             aurgment=False)

    val_gen = DatasetGeneratorFromCocoJson(batch_size=batch_size, image_list=images_valid, coco=coco_valid,
                                           path_folder=os.path.join(DATASET_PATH, 'valid'), classes=classes_train,
                                           aurgment=False)

    img, mask = train_gen.__getitem__(0)
    gen_viz(img_s=img, mask_s=mask)
    # return 0
    # visualizeImageOrGenerator(images_list=img, mask_list=mask)
    #
    # img1, mask1 = val_gen.__getitem__(0)
    # visualizeImageOrGenerator(images_list=img1, mask_list=mask1)

    # return 0
    model = unet(input_shape=(128, 128, 3), num_classes=len(classes_train))

    tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # return 0
    path_model = os.path.join(MODEL_H5_PATH, MODEL_H5_FILE_NAME)
    model_history = get_last_model_history()
    if model_history:
        path_model = os.path.join(MODEL_H5_PATH, model_history.name_file)
    print(path_model)

    history = train_model(path_model=path_model,
                          model=model,
                          n_epoch=1000,
                          batch_size=batch_size,
                          dataset_train=train_gen,
                          dataset_valid=val_gen,
                          dataset_size_train=len(images_train),
                          # dataset_size_val=len(images_valid),
                          model_history=model_history,
                          monitor='loss',
                          mode='min'
                          )

    plot_segm_history(history, metrics=['my_mean_iou', 'val_my_mean_iou'])

    path_zip = zipfile.ZipFile(f'{os.path.splitext(path_model)[0]}.zip', 'w')
    path_zip.write(path_model, arcname=f'{model_history.name_file}')
    path_zip.close()

    model.save(path_model)

    if model_history:
        model_history.date_train = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        model_history.quality_dataset = len(images_train) + len(images_train)
        model_history.quality_valid_dataset = str(len(images_train))
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
