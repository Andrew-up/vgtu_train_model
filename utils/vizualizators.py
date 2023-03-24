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

from definitions import ANNOTATION_FILE_PATH, DATASET_PATH, MODEL_H5_PATH, ANNOTATION_FILE_PATH_TRAIN, \
    ANNOTATION_FILE_PATH_TEST, ANNOTATION_FILE_PATH_VALID
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.model_losses import dice_coef, bce_dice_loss, jaccard_distance, iou, jaccard_coef, dice_coef_loss, \
    binary_weighted_cross_entropy, dice_loss


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def vizualizator(list_images, list_masks, classes):
    x, y = list_images, list_masks
    fig = plt.figure(figsize=(50, 40))
    gs = gridspec.GridSpec(nrows=len(x), ncols=2)
    colors = ['#705335', '#25221B', '#E63244', '#EC7C26', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    labels = classes
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]


    flag = False
    for i in range(0, len(list_images)):

        images, mask = x[i], y[i]
        sample_img = images
        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img)

        ax1 = fig.add_subplot(gs[i, 1])
        if (flag == False):
            flag = True
            ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)
            ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize=14,
                       title='Mask Labels', title_fontsize=14, edgecolor="black", facecolor='#c5c6c7')

        l0 = ax1.imshow(sample_img)
        listMasks = []
        for i in range(len(mask[0, 0, :])):
            mask_one = mask[:, :, i]
            l = ax1.imshow(np.ma.masked_where(
                mask_one == False, mask_one), cmap=mpl.colors.ListedColormap(colors[i]), alpha=1)
            listMasks.append(l)

        colors = [im.cmap(im.norm(1)) for im in listMasks]

    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.3,
                        top=0.92, wspace=0.01, hspace=0.08)
    plt.show()


from keras import backend as K


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH,
                                                              percent_valid=0,
                                                              # path_folder='test'
                                                              )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_10.h5')

    iou1111 = MyMeanIOU(num_classes=len(classes),
                        ignore_class=0
                        )
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss,
                                                'MyMeanIOU': iou1111,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })
    # model = load_model(model)
    for j in range(1):
        train_generator_class = DataGeneratorFromCocoJson(batch_size=6,
                                                          # path_folder='test',
                                                          subset='train',
                                                          input_image_size=(128, 128),
                                                          image_list=images_train,
                                                          classes=classes,
                                                          coco=coco,
                                                          shuffle=False)

        img_s, mask_s = train_generator_class.__getitem__(0)
        res = model.predict(img_s)

        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=len(img_s), ncols=4)
        colors = ['#0000ff', '#ff0066', '#66ff33', '#ffff00', '#ffccff', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754', '#6C4675', '#969992', '#9E9764', '#969992', '#9E9764']
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
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss,
                                                'MyMeanIOU': iou1111,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })

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
    pppppppppp()
    # main()
    # viz_model()
    show_mask_true_and_predict()
