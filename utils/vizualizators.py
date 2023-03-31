import os.path
import random

import keras.utils
from matplotlib import gridspec, pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from keras.models import Model, load_model
import tensorflow as tf
from keras.utils import img_to_array

from definitions import ANNOTATION_FILE_PATH, DATASET_PATH, MODEL_H5_PATH, ANNOTATION_FILE_PATH_TRAIN
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.newDataGeneratorCoco import cocoDataGenerator, augmentationsGenerator, visualizeGenerator
from utils.get_dataset_coco import filterDataset
from utils.model_losses import bce_dice_loss, \
    binary_weighted_cross_entropy, dice_loss


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def visualize_mask(mask):
    print(mask.shape)

    # Определяем цвета для каждого класса
    colors = {
        0: (0, 0, 0),  # Черный цвет для фона
        1: (255, 0, 0),  # Красный цвет для класса 1
        2: (0, 255, 0),  # Зеленый цвет для класса 2
        3: (0, 0, 255)  # Синий цвет для класса 3
    }

    # Создаем цветовую маску
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            class_id = int(mask[i, j, 0])
            if class_id in colors:
                color_mask[i, j, :] = colors[class_id]

    # Выводим цветовую маску
    plt.imshow(color_mask)
    plt.axis('off')
    plt.show()


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


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


def visualize_mask2(mask):
    # Создаем пустой массив для раскрашенной маски
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
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


def vizualizator(gen):
    x, y = next(gen)
    for i in range(len(x)):
        image = x[i]
        mask = y[i]
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.axis('off')

        color_mask111 = visualize_mask2(mask=mask)
        ax2.imshow(color_mask111)
        ax2.axis('off')
        plt.show()


from keras import backend as K


def dice_loss1111(y_true, y_pred, smooth=1):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return 1 - ((numerator + smooth) / (denominator + smooth))


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )

    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_19.h5')

    iou1111 = MyMeanIOU(num_classes=len(classes),
                        # ignore_class=0
                        )

    model = load_model(paths_m, custom_objects={
        'MyMeanIOU': iou1111,
        'dice_loss': dice_loss1111,
    })

    for j in range(1):
        train_generator_class = cocoDataGenerator(images_train,
                                                  classes=classes,
                                                  coco=coco,
                                                  mask_type="normal",
                                                  input_image_size=(128, 128),
                                                  batch_size=4)

        img_s, mask_s = next(train_generator_class)
        res = model.predict(img_s)
        for i in range(len(res)):
            for j in range(len(res[i, 0, 0, :])):

                plt.imshow(res[i, :, :, j], alpha=0.4)
            plt.show()
        print(res.shape)


palette = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (0, 0, 0),
    6: (0, 0, 0),
    7: (0, 0, 0),
    8: (0, 0, 0),
    9: (255, 0, 0),
    10: (0, 0, 255),
    11: (0, 255, 0)
}


def mask2img(mask):
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[np.argmax(mask[j, i])]
    return image


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def pppppppppp():
    # return 0
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_10.h5')

    iou1111 = MyMeanIOU(num_classes=len(classes)
                        # , ignore_class=0
                        )
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss1111,
                                                'MyMeanIOU': iou1111,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })

    # train_generator_class = DataGeneratorFromCocoJson(batch_size=6,
    #                                                   # path_folder='test',
    #                                                   subset='train',
    #                                                   input_image_size=(128, 128),
    #                                                   image_list=images_train,
    #                                                   classes=classes,
    #                                                   coco=coco,
    #                                                   shuffle=True)  #
    train_generator_class = DataGeneratorFromCocoJson(batch_size=6,
                                                      # path_folder='test',
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=True)
    from tensorflow import expand_dims

    img_s, mask_s = train_generator_class.__getitem__(0)
    # predddd = model.predict(img_s)
    # m = tf.keras.metrics.MeanIoU(num_classes=5 - 1)
    # m.update_state(y_true=mask_s, y_pred=predddd)
    # print('result iou: ' + str(m.result().numpy()))

    for i in range(len(img_s)):
        img_one = img_s[i]
        img_array_batch = expand_dims(img_one, axis=0)  # Create a batch
        pre = model.predict(img_array_batch)

        rows, cols = 5, 5
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')
        c_count = 0
        r_count = 0
        num_class_model = len(pre[0, 0, 0, :])
        # d = classes

        for J in range(num_class_model):
            if c_count == cols:
                r_count += 1
                c_count = 0
            # ax[r_count, c_count].imshow(np.array(pre[0, :, :, i].astype(np.float32) > 0.9))
            ax[r_count, c_count].imshow(np.array(pre[0, :, :, J].astype(np.float32) > 0.5))
            ax[r_count, c_count].set_title(classes[J])
            ax[r_count, c_count].set_xticks(())
            ax[r_count, c_count].set_yticks(())
            c_count += 1
        ax[4, 0].set_title('original')
        ax[4, 0].imshow(img_one)
        ax[4, 0].set_xticks(())
        ax[4, 0].set_xticks(())

        plt.show()


def viz_model():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=0)
    model = load_model(MODEL_H5_PATH, custom_objects={'dice_coef': dice_coef,
                                                      'bce_dice_loss': bce_dice_loss})

    train_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)

    # m = tf.keras.metrics.IoU()
    # m.update_state(train_generator_class, Y_pred)
    # print(m.result().numpy())

    # my = MyMeanIOU(num_classes=3)
    # my.update_state(mask_s, Y_pred)
    # print(my.result().numpy())

    # d_coef = dice_coef(mask_s.astype('float32'), Y_pred)
    # print(d_coef)
    # sc = np.squeeze(np.round(Y_pred))
    # (np.squeeze(np.round(Y_pred)) == train_generator_class).mean()
    # print((np.squeeze(np.round(Y_pred)) == train_generator_class).mean())
    # z = dice_coef(train_generator_class.astype('float32'), pred).numpy()
    # print(z)
    # return 0
    # z = dice_coef(train_generator_class.astype('float32'), pred).numpy()
    # print(z)
    image1 = train_generator_class.getImage(DATASET_PATH + '/0001.png')
    # plt.imshow(img_s[0])
    # plt.show()
    # img_array = img_to_array(img_s[0, :, :, :])
    img_array = img_to_array(image1)
    img_array_batch = tf.expand_dims(img_array, axis=0)  # Create a batch
    # img_array_batch /= 255.
    # res = model.predict(img_array_batch)
    # print(res.shape)
    # # return 0
    # plt.figure()
    # print(np.sum(res[0, :, :, 0]))
    # print(np.sum(res[0, :, :, 1]))
    # print(np.sum(res[0, :, :, 2]))
    # list_predict = list()
    # list_predict.append(np.sum(res[0, :, :, 0]))
    # list_predict.append(np.sum(res[0, :, :, 1]))
    # list_predict.append(np.sum(res[0, :, :, 2]))
    # max_value = max(list_predict)
    # max_index = list_predict.index(max_value)
    # # print(max_value)
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(res[0, :, :, i], cmap='viridis')
    #     plt.title('class:' + str(i+1))
    #     if i == max_index:
    #         plt.title('MAX activation:' + str(i+1))
    #     plt.axis('off')
    # plt.show()

    # return 0
    print(model.summary())
    activation_model = Model(inputs=model.input, outputs=model.layers[52].output)
    activation_model.summary()
    activation = activation_model.predict(img_array_batch)
    images_per_row = 16
    n_filters = activation.shape[-1]
    size = activation.shape[1]
    n_cols = n_filters // images_per_row
    display_grid = np.zeros((n_cols * size, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def test():
    listnumpy = list()
    a = np.array([[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]], [[0, 3], [3, 3], [3, 3]]])
    # a = np.random.rand(3, 3, 2)
    print(a)
    # print(a[:, :, 0])
    print('----------------------')
    # print(a[:, :, 1])
    print('----------------------')
    print(np.sum(a[:, :, 0]))
    listnumpy.append(np.sum(a[:, :, 0]))
    listnumpy.append(np.sum(a[:, :, 1]))
    # listnumpy.append(np.sum(a[:, :, 0])
    print(listnumpy)

    print('----------------------')
    print('max index: ')
    max_value = max(listnumpy)
    print(listnumpy.index(max_value))
    print()
    # print(a[:, :, 1])
    # print('----------------------')
    # print(a[:, :, 3])
    # print(np.argmax(a, axis=1))
    # print(np.argmax(a, axis=2))
    # print(a)
    print(a.shape)
    # print(np.sum(a[0, :, 0]))
    # print(np.sum(a, axis=1))
    # print(np.sum(a, axis=2))


if __name__ == "__main__":
    #     # test()
    # pppppppppp()
    # main()
    # viz_model()
    show_mask_true_and_predict()
