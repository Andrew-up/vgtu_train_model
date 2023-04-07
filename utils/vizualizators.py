import os.path

import keras.backend as K
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from matplotlib import gridspec, pyplot as plt

from definitions import ANNOTATION_FILE_PATH, MODEL_H5_PATH, DATASET_PATH
from utils.CocoGenerator_new import DatasetGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.newDataGeneratorCoco import cocoDataGenerator, augmentationsGenerator

colors = [
    [255, 255, 255],  # фон
    [0, 255, 0],  # Зеленый
    [0, 0, 255],  # Синий
    [255, 255, 0]  # Желтый
]

def create_mask(pred_mask):
 pred_mask = np.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask

def color_mask(mask_ggggg):
    mask = mask_ggggg
    print(mask_ggggg.shape)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Получаем класс текущего пикселя
            cls = mask[i, j, 0]
            # Получаем цвет для данного класса из словаря
            color = colors[cls]
            # Раскрашиваем пиксель в соответствующий цвет
            colored_mask[i, j, :] = color
    return colored_mask

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[..., 1:])  # удаляем первый канал, отвечающий за фон
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def viz_model():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=0, shuffie=False)
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


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ann_file_name='_annotations.coco.json', percent_valid=0, shuffie=False, path_folder='train')
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_19.h5')
    dddd = MyMeanIOU(num_classes=3)
    model = load_model(paths_m, custom_objects={
        'dice_coef': dice_coef,
        'MyMeanIOU': dddd,
        # 'dice_coef': dice_coef,
        # 'jaccard_coef': jaccard_coef
    })
    train_gen = DatasetGeneratorFromCocoJson(batch_size=4, image_list=images_train, coco=coco,
                                             path_folder=os.path.join(DATASET_PATH, 'train'), classes=classes, aurgment=True)
    for i in range(10):
        img, mask = train_gen.__getitem__(i)
        pre = model.predict(img)
        fig, ax = plt.subplots(ncols=3, nrows=4)
        ax[0][0].title.set_text('original image')
        ax[0][1].title.set_text('original mask')
        ax[0][2].title.set_text('predict mask image')

        # y_pred3 = np.argmax(y_pred[i], axis=-1)


        for i in range(4):
            # y_pred3 = np.argmax(y_pred[i], axis=-1)
            ax[i][0].imshow(img[i])
            ax[i][1].imshow(color_mask(create_mask(mask[i])))
            # jjjjjjjj = pre[i, :, :, :]
            predmask = color_mask(create_mask(pre[i]))
            ax[i][2].imshow(predmask)
            # ax[i][0].axis('off')
            # ax[i][1].axis('off')
            # ax[i][2].axis('off')

        plt.show()

def gen_viz(img_s, mask_s, pred = None):
    print(img_s.shape)
    print(mask_s.shape)
    fig = plt.figure(figsize=(10, 25))
    gs = gridspec.GridSpec(nrows=len(img_s), ncols=3)
    colors = ['yellow', 'green', 'red']
    labels = ["Small Bowel", "Large Bowel", "Stomach"]
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3 = mpl.colors.ListedColormap(colors[2])
    flag = False
    for i in range(0, 4):

        images, mask = img_s[i], mask_s[i]
        sample_img = images / 255.
        mask1 = mask[:, :, 0]
        mask2 = mask[:, :, 1]
        mask3 = mask[:, :, 2]

        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img[:, :, 0], cmap='gray')
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])
        if (flag == False):
            flag = True
            ax0.set_title("Image", fontsize=15, weight='bold', y=1.02)
            ax1.set_title("Mask", fontsize=15, weight='bold', y=1.02)
            if pred is not None:

                ax2.set_title("predicted Mask", fontsize=15, weight='bold', y=1.02)
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4, fontsize=14,
                       title='Mask Labels', title_fontsize=14, edgecolor="black", facecolor='#c5c6c7')

        l0 = ax1.imshow(sample_img[:, :, 0], cmap='gray')
        l1 = ax1.imshow(np.ma.masked_where(
            mask1 == False, mask1), cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(
            mask2 == False, mask2), cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(
            mask3 == False, mask3), cmap=cmap3, alpha=1)

        if pred is not None:

            pre = pred[i]
            predict1 = pre[:, :, 0]
            predict1 = (predict1 > 0.5).astype(np.float32)
            predict1 = np.array(predict1)
            predict2 = pre[:, :, 1]
            predict2 = (predict2 > 0.5).astype(np.float32)
            predict2 = np.array(predict2)
            predict3 = pre[:, :, 2]
            predict3 = (predict3 > 0.5).astype(np.float32)
            predict3 = np.array(predict3)
            l0 = ax2.imshow(sample_img[:, :, 0], cmap='gray')
            l1 = ax2.imshow(np.ma.masked_where(
                predict1 == False, predict1), cmap=cmap1, alpha=1)
            l2 = ax2.imshow(np.ma.masked_where(
                predict2 == False, predict2), cmap=cmap2, alpha=1)
            l3 = ax2.imshow(np.ma.masked_where(
                predict3 == False, predict3), cmap=cmap3, alpha=1)
            _ = [ax.set_axis_off() for ax in [ax0, ax1]]


        colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]

    plt.show()
if __name__ == "__main__":
    #     # test()
    # pppppppppp()
    # main()
    # viz_model()
    # show_mask_true_and_predict()
    show_mask_true_and_predict()
