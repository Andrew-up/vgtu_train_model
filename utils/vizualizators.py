import os.path

import keras.backend as K
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from matplotlib import gridspec, pyplot as plt

from definitions import ANNOTATION_FILE_PATH, MODEL_H5_PATH
from utils.get_dataset_coco import filterDataset
from utils.newDataGeneratorCoco import cocoDataGenerator, augmentationsGenerator


class MeanDiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name='mean_dice_coefficient', **kwargs):
        super(MeanDiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice_coefficient = self.add_weight(name='dice_coefficient', initializer='zeros', shape=())

    def update_state(self, y_true, y_pred, sample_weight=None):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
        dice = numerator / denominator
        self.dice_coefficient.assign_add(tf.reduce_mean(dice))

    def result(self):
        return self.dice_coefficient

    def reset_state(self):
        self.dice_coefficient.assign(0.0)


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    dice = numerator / denominator
    loss = 1 - tf.reduce_mean(dice)
    return loss


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


color_map__tf = tf.constant([
    [0, 0, 0],  # Канал 0 (фон) - черный цвет
    [255, 0, 0],  # Канал 1 - красный цвет
    [0, 255, 0],  # Канал 2 - зеленый цвет
    [0, 0, 255],  # Канал 3 - синий цвет
    [255, 255, 0],  # Канал 4 - желтый цвет
    [255, 0, 255],  # Канал 5 - пурпурный цвет
    [0, 255, 255],  # Канал 6 - голубой цвет
    [128, 0, 0],  # Канал 7 - темно-красный цвет
    [0, 128, 0],  # Канал 8 - темно-зеленый цвет
    [0, 0, 128],  # Канал 9 - темно-синий цвет
    [128, 128, 0],  # Канал 10 - темно-желтый цвет
    [128, 0, 128],  # Канал 11 - темно-пурпурный цвет
], dtype=tf.uint8)


def show_mask_true_and_predict_old():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_20.h5')

    iou1111 = MyMeanIOU(num_classes=len(classes),
                        # ignore_class=0
                        )
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss,
                                                'MyMeanIOU': iou1111,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })
    # model = load_model(model)
    for j in range(1):
        train_generator_class = cocoDataGenerator(images_train,
                                                  classes=classes,
                                                  coco=coco,
                                                  mask_type="normal",
                                                  input_image_size=(128, 128),
                                                  batch_size=4)

        aug_gen = augmentationsGenerator(train_generator_class)

        img_s, mask_s = next(train_generator_class)
        res = model.predict(img_s)
        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=len(img_s), ncols=4)
        colors = ['#ffccff', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754', '#6C4675', '#969992', '#9E9764', '#0000ff',
                  '#ff0066', '#66ff33', '#ffff00', '#969992', '#9E9764', ]
        patches = [mpatches.Patch(
            color=colors[i], label=f"{classes[i]}") for i in range(len(classes))]

        flag = False
        for i in range(0, len(img_s)):
            images, mask = img_s[i], mask_s[i]
            sample_img = images
            ax0 = fig.add_subplot(gs[i, 0])
            im = ax0.imshow((sample_img * 255).astype(np.uint8))
            ax1 = fig.add_subplot(gs[i, 1])
            ax2 = fig.add_subplot(gs[i, 2])
            if (flag == False):
                flag = True
                ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)
                ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)
                ax2.set_title("predicted Mask", fontsize=15, weight='bold', y=1.02)
                plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize=14,
                           title='Mask Labels', title_fontsize=14, edgecolor="black", facecolor='#c5c6c7')

            l0 = ax1.imshow(sample_img)
            listMasks = []
            listMasksPredict = []
            l0 = ax2.imshow(sample_img)
            for m in range(len(mask[0, 0, :])):
                mask_one = mask[:, :, m]
                l = ax1.imshow(np.ma.masked_where(
                    mask_one == False, mask_one), cmap=mpl.colors.ListedColormap(colors[m]), alpha=1)
                listMasks.append(l)

            pre = res[i]
            print(pre.shape)
            for i in range(len(pre[0, 0, :])):
                # print(i)
                res_one = (pre[:, :, i] > 0.5).astype(np.float32)
                res_one = np.array(res_one)
                pass
                l = ax2.imshow(np.ma.masked_where(
                    res_one == False, res_one), cmap=mpl.colors.ListedColormap(colors[i]), alpha=1)
                listMasksPredict.append(l)

            _ = [ax.set_axis_off() for ax in [ax0, ax1]]

            colors = [im.cmap(im.norm(1)) for im in listMasks]
            colors2 = [im.cmap(im.norm(1)) for im in listMasksPredict]
            # colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]]

        plt.show()


class IOU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=3, name=None, dtype=None):
        super(IOU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super(IOU, self).update_state(y_true, y_pred, sample_weight)


colors_dict = {
    0: [0, 0, 0],  # фон
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
    14: [0, 0, 0]  # Черный
}


def color_mask(mask):
    # Создаем пустой массив для раскрашенной маски
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Создаем словарь с цветами для каждого класса
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Получаем класс текущего пикселя
            cls = mask[i, j, 0]
            # Получаем цвет для данного класса из словаря
            color = colors_dict[cls]
            # Раскрашиваем пиксель в соответствующий цвет
            colored_mask[i, j, :] = color
    return colored_mask


def vizualizator_old(gen, classes):
    x, y = next(gen)
    fig = plt.figure(figsize=(50, 40))
    gs = gridspec.GridSpec(nrows=len(x), ncols=2)
    colors = ['#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    labels = classes
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    flag = False
    for i in range(0, len(x)):

        images, mask = x[i], y[i]
        sample_img = images
        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img)

        ax1 = fig.add_subplot(gs[i, 1])
        if (flag == False):
            flag = True
            ax0.set_title("Image", fontsize=25, weight='bold', y=1.02)
            ax1.set_title("Mask", fontsize=25, weight='bold', y=1.02)
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize=14,
                       title='Mask Labels', title_fontsize=14, edgecolor="black", facecolor='#c5c6c7')

        ax1.imshow(color_mask(mask))


    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.3,
                        top=0.92, wspace=0.01, hspace=0.08)
    plt.show()


def show_mask_true_and_predict_old2():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_19.h5')

    dddd = MyMeanIOU(num_classes=3)
    model = load_model(paths_m, custom_objects={
        'dice_loss': dice_loss,
        'MyMeanIOU': dddd,
        # 'dice_coef': dice_coef,
        # 'jaccard_coef': jaccard_coef
    })

    # model = load_model(model)
    for j in range(1):
        train_generator_class = cocoDataGenerator(images_train,
                                                  classes=classes,
                                                  coco=coco,
                                                  mask_type="normal",
                                                  input_image_size=(128, 128),
                                                  batch_size=4)

        aug_gen = augmentationsGenerator(train_generator_class)

        img_s, mask_s = next(aug_gen)
        res = model.predict(img_s)
        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=len(img_s), ncols=4)
        colors = ['#ffccff', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754', '#6C4675', '#969992', '#9E9764', '#0000ff',
                  '#ff0066', '#66ff33', '#ffff00', '#969992', '#9E9764', ]
        patches = [mpatches.Patch(
            color=colors[i], label=f"{classes[i]}") for i in range(len(classes))]

        flag = False
        for i in range(0, len(img_s)):
            images, mask = img_s[i], mask_s[i]
            sample_img = images
            ax0 = fig.add_subplot(gs[i, 0])
            im = ax0.imshow((sample_img * 255).astype(np.uint8))
            ax1 = fig.add_subplot(gs[i, 1])
            ax2 = fig.add_subplot(gs[i, 2])
            if (flag == False):
                flag = True
                ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)
                ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)
                ax2.set_title("predicted Mask", fontsize=15, weight='bold', y=1.02)
                plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize=14,
                           title='Mask Labels', title_fontsize=14, edgecolor="black", facecolor='#c5c6c7')

            l0 = ax1.imshow(sample_img)
            listMasks = []
            listMasksPredict = []
            # l0 = ax2.imshow(sample_img)
            for m in range(len(mask[0, 0, :])):
                mask_one = mask[:, :, m]
                l = ax1.imshow(np.ma.masked_where(
                    mask_one == False, mask_one), cmap=mpl.colors.ListedColormap(colors[m]), alpha=1)
                listMasks.append(l)

            pre = res[i]
            print(pre.shape)
            for hhhh in range(len(pre[0, 0, :])):
                print(hhhh)
                # print(i)
                res_one = (pre[:, :, hhhh] > 0.3).astype(np.float32)
                res_one = np.array(res_one)
                pass

                l = ax2.imshow(np.ma.masked_where(
                    res_one == False, res_one), cmap=mpl.colors.ListedColormap(colors[hhhh]), alpha=1)
                listMasksPredict.append(l)

            _ = [ax.set_axis_off() for ax in [ax0, ax1]]

            colors = [im.cmap(im.norm(1)) for im in listMasks]
            colors2 = [im.cmap(im.norm(1)) for im in listMasksPredict]
            # colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]]

        plt.show()


def show_images_with_masks(images, masks, predicted_masks=None):
    num_images = len(images)
    fig, ax = plt.subplots(nrows=num_images, ncols=3, figsize=(15, 15))
    ax = ax.ravel()

    # Отображаем изображения, оригинальные маски и предсказанные маски
    for i in range(num_images):
        ax[i * 3].imshow(images[i])
        ax[i * 3].set_title('Оригинальное изображение')
        ax[i * 3].axis('off')
        mask_c = np.zeros_like(images[i])
        colored_mask = tf.squeeze(tf.gather(color_map__tf, masks[i]), axis=-2)
        # mmmmm =
        # mask_c[color_mask(mmmmm) == 0] = 1
        ax[i * 3 + 1].imshow(colored_mask, alpha=1)
        # ax[i * 3 + 1].imshow(mask_c, alpha=1)
        ax[i * 3 + 1].set_title('Оригинальная маска')
        ax[i * 3 + 1].axis('off')

        if predicted_masks is not None:
            print(predicted_masks[i].shape)
            mask_1_channel = semantic_segmentation(predicted_masks[i])
            mask_c2 = np.zeros_like(images[i])
            mask_c2[color_mask(mask_1_channel) == 0] = 1
            ax[i * 3 + 2].imshow(images[i], alpha=1)
            ax[i * 3 + 2].imshow(mask_c2, alpha=1)
            ax[i * 3 + 2].set_title('Предсказанная маска')
            ax[i * 3 + 2].axis('off')

    plt.tight_layout()
    plt.show()


def vizualizator(gen):
    x, y = next(gen)
    show_images_with_masks(images=np.array(x), masks=np.array(y))


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )

    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_20.h5')

    iou1111 = MyMeanIOU(num_classes=len(classes),
                        # ignore_class=0
                        )

    model = load_model(paths_m, custom_objects={
        'dice_loss': dice_loss,
        'MeanDiceCoefficient': MeanDiceCoefficient(),
    })

    train_generator_class = cocoDataGenerator(images_train,
                                              classes=classes,
                                              coco=coco,
                                              mask_type="normal",
                                              input_image_size=(128, 128),
                                              batch_size=4)

    img_s, mask_s = next(train_generator_class)
    # mask_one = mask_s[0]
    res = model.predict(img_s)

    show_images_with_masks(images=img_s, masks=mask_s, predicted_masks=res)
    # display([img_s[0], gt_mask, predicted])
    # print('2')
    # cv2.imshow('Result', img_s[0])
    # cv2.waitKey(0)


def semantic_segmentation(mask):
    # получаем индекс канала с максимальным значением в каждом пикселе
    mask_1_channel = np.argmax(mask, axis=-1, keepdims=True)
    return mask_1_channel


def jaccard_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    iou = (intersection + 1e-15) / (sum_ - intersection + 1e-15)
    return 1 - iou


def pppppppppp():
    # return 0
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_20.h5')

    dddd = MyMeanIOU(num_classes=3)
    model = load_model(paths_m, custom_objects={
        'dice_loss': dice_loss,
        'MyMeanIOU': dddd,
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

    train_generator_class = cocoDataGenerator(images_train,
                                              classes=classes,
                                              coco=coco,
                                              mask_type="normal",
                                              input_image_size=(128, 128),
                                              batch_size=4)
    from tensorflow import expand_dims

    img_s, mask_s = next(train_generator_class)
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
        print(num_class_model)
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


def mean_iou(y_true, y_pred):
    num_classes = K.int_shape(y_pred)[-1]
    iou = []
    for c in range(num_classes):
        true_class = y_true[..., c]
        pred_class = y_pred[..., c]
        intersection = K.sum(true_class * pred_class)
        union = K.sum(true_class) + K.sum(pred_class) - intersection
        iou.append((intersection + K.epsilon()) / (union + K.epsilon()))
    return K.mean(K.stack(iou))


def viz_model():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=0, shuffie=False)
    iou1111 = MyMeanIOU(num_classes=len(classes)
                        # , ignore_class=0
                        )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_20.h5')
    dddd = MyMeanIOU(num_classes=3)
    model = load_model(paths_m, custom_objects={
        'dice_loss': dice_loss,
        'MyMeanIOU': dddd,
        # 'dice_coef': dice_coef,
        # 'jaccard_coef': jaccard_coef
    })

    train_generator_class = cocoDataGenerator(images_train,
                                              classes=classes,
                                              coco=coco,
                                              mask_type="normal",
                                              input_image_size=(128, 128),
                                              batch_size=4)
    from tensorflow import expand_dims

    img_s, mask_s = next(train_generator_class)
    img_array_batch = expand_dims(img_s[0], axis=0)  # Create a batch
    # return 0
    print(model.summary())
    # return 0
    activation_model = Model(inputs=model.input, outputs=model.layers[70].output)
    activation_model.summary()
    activation = activation_model.predict(img_array_batch)
    print('1111111111')
    images_per_row = 3
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


# def color_mask(mask):
#     # Создаем пустой массив для раскрашенной маски
#     # print('1111111111111111')
#     colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#     # Создаем словарь с цветами для каждого класса
#     colors_dict = {
#         0: [0, 0, 0],  # Черный
#         1: [255, 255, 255],  # Белый
#         2: [255, 0, 0],  # Красный
#         3: [0, 255, 0],  # Зеленый
#         4: [0, 0, 255],  # Синий
#         5: [255, 255, 0],  # Желтый
#         6: [255, 0, 255],  # Розовый
#         7: [0, 255, 255],  # Бирюзовый
#         8: [128, 0, 0],  # Темно-красный
#         9: [0, 128, 0],  # Темно-зеленый
#         10: [0, 0, 128],  # Темно-синий
#         11: [128, 128, 128]  # Серый
#     }
#     # print(mask.shape)
#     # Проходимся по всем пикселям маски
#     for i in range(mask.shape[0]):
#         for j in range(mask.shape[1]):
#             # Получаем класс текущего пикселя
#             cls = mask[i, j, 0]
#             # print(cls)
#             # Получаем цвет для данного класса из словаря
#             color = colors_dict[cls]
#             # Раскрашиваем пиксель в соответствующий цвет
#             colored_mask[i, j, :] = color
#     return colored_mask


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
    # show_mask_true_and_predict()
    show_mask_true_and_predict_old2()
