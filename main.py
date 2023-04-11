import os.path
import time
import zipfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history
from definitions import MODEL_H5_PATH, MODEL_H5_FILE_NAME, DATASET_PATH
from utils.CocoGenerator_new import DatasetGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.helpers import delete_legacy_models_and_zip
from utils.model_losses import plot_segm_history
from utils.model_train import train_model
from utils.unet import unet, build_unet, jaccard_index
from utils.vizualizators import gen_viz


def main():
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=10)
    if check_garbage_files_count == 0:
        print('Мусора нет')
    else:
        print(f'Удалено старых моделей h5 и zip архивов: {check_garbage_files_count}')
    timer = time.time()
    train_path = 'train'
    valid_path = 'valid'

    images_train, _, coco_train, classes_train = filterDataset(ann_file_name='labels_my-project-name_2022-11-15-02-32-33.json',
                                                               percent_valid=0,
                                                               path_folder=train_path
                                                               )

    # return 0
    images_valid, _, coco_valid, classes_valid = filterDataset(ann_file_name='labels_my-project-name_2022-11-15-02-32-33.json',
                                                               percent_valid=0,
                                                               path_folder=valid_path
                                                               )

    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ТРЕНИРОВКИ: {len(images_train)}')
    print(f'РАЗМЕР ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ: {len(images_valid)}')

    print(classes_train)
    input_image_size = (128, 128)
    batch_size = 8
    # train_gen = cocoDataGenerator(
    #     images_train,
    #     classes_train,
    #     coco_train,
    #     folder='train',
    #     mask_type="normal",
    #     input_image_size=(128, 128),
    #     batch_size=batch_size,
    # )
    # val_gen = cocoDataGenerator(
    #     images_train,
    #     classes_train,
    #     coco_train,
    #     folder='train',
    #     mask_type="normal",
    #     input_image_size=(128, 128),
    #     batch_size=batch_size,
    # )

    train_gen = DatasetGeneratorFromCocoJson(batch_size=batch_size, image_list=images_train, coco=coco_train,
                                             path_folder=os.path.join(DATASET_PATH, train_path), classes=classes_train,
                                             aurgment=False, input_image_size=input_image_size)

    val_gen = DatasetGeneratorFromCocoJson(batch_size=batch_size, image_list=images_valid, coco=coco_valid,
                                           path_folder=os.path.join(DATASET_PATH, valid_path), classes=classes_valid,
                                           aurgment=False, input_image_size=input_image_size)
    # for j in range(5):
    #     img, mask = next(train_gen)
    #     gen_viz(img_s=img, mask_s=mask, epoch='start train test mask')
    # return 0

        # print(img.shape)

    # return 0


    # visualizeGenerator(gen=None, img=img, pred=mask)
    # visualizeGenerator(val_gen)



    # return 0

    img, mask = val_gen.__getitem__(0)
    gen_viz(img_s=img, mask_s=mask)



    # return 0

    # return 0
    # gen_viz(img_s=img, mask_s=mask)
    # return 0

    # fig1, axs1 = plt.subplots(nrows=len(mask[:, 0, 0, 0]), ncols=4, figsize=(5, 5))
    # fig1.tight_layout()
    # for i in range(mask.shape[0]):
    #     axs1[i][3].imshow(img[i, :, :, :])
    #     for j in range(mask.shape[-1]):
    #         axs1[i][j].imshow(mask[i, :, :, j])
    #         axs1[i][j].set_title(f'Class {j}')
    #         axs1[i][j].axis('off')
    # plt.show()

    # return 0

    # return 0

    # return 0
    # visualizeImageOrGenerator(images_list=img, mask_list=mask)
    #
    # img1, mask1 = val_gen.__getitem__(0)
    # visualizeImageOrGenerator(images_list=img1, mask_list=mask1)

    # return 0
    model = unet(input_shape=(input_image_size[0], input_image_size[1], 3), num_classes=len(classes_train))
    # model = build_unet(input_shape=(input_image_size[0], input_image_size[1], 3), num_classes=len(classes_train))
    # model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # return 0
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
                          dataset_valid=val_gen,
                          dataset_size_train=len(images_train),
                          dataset_size_val=len(images_valid),
                          model_history=model_history,
                          monitor='loss',
                          mode='min'
                          )

    plot_segm_history(history, metrics=['jaccard_index', 'val_jaccard_index'])

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
