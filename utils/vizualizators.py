import os.path
from enum import Enum

import keras.backend as K
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from matplotlib import gridspec, pyplot as plt
from segmentation_models.losses import cce_dice_loss

from definitions import ANNOTATION_FILE_PATH, MODEL_H5_PATH, DATASET_PATH
from utils.CocoGenerator_new import DatasetGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.newDataGeneratorCoco import cocoDataGenerator, augmentationsGenerator


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def display(img=None, mask=None, pred=None):
    size = 0
    if img is not None:
        size += 1
    if mask is not None:
        size += 1
    if pred is not None:
        size += 1
    plt.figure(figsize=(10, 10))
    title = ['image original', 'original Mask', 'predicted_mask']
    # print(len(display_list[0][:, 0, 0, 0]))
    fig, axs = plt.subplots(nrows=len(img),
                            ncols=size,
                            figsize=(5, 5))
    axs[0][0].set_title(title[0])
    axs[0][1].set_title(title[1])
    if pred is not None:
        axs[0][2].set_title(title[2])
    for j in range(len(img)):
        for i in range(size):
            axs[j][i].axis('off')
            image = np.zeros((128, 128), dtype=np.uint8)
            if i == 0:
                image = tf.keras.preprocessing.image.array_to_img(img[j])
            if i == 1:
                mask_one = create_mask(mask[j] > 0.7)
                image = tf.keras.preprocessing.image.array_to_img(mask_one)
            if i == 2:
                pred_one = create_mask(pred[j] > 0.2)
                image = tf.keras.preprocessing.image.array_to_img(pred_one)
            axs[j][i].imshow(image)
    plt.show()


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

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred) - tf.reduce_sum(y_true * y_pred)
    return 1 - numerator / denominator


colors = np.array([
    [0, 0, 0],    # Цвет для класса 0
    [255, 0, 0],  # Цвет для класса 1
    [0, 255, 0],  # Цвет для класса 2
    [0, 0, 255],  # Цвет для класса 3
    [255, 255, 0] # Цвет для класса 4
])


def show_mask_true_and_predict():
    images_train, _, coco_train, classes_train = filterDataset(ann_file_name='labels_my-project-name_2022-11-15-02-32-33.json',
                                                               percent_valid=0,
                                                               path_folder='train'
                                                               )
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_19.h5')
    dddd = MyMeanIOU(num_classes=3)
    model = load_model(paths_m, custom_objects={
        'categorical_crossentropy_plus_dice_loss': cce_dice_loss,
        'MyMeanIOU': dddd,
        # 'dice_loss': dice_loss,
        # 'jaccard_coef': jaccard_coef
    })
    train_gen = DatasetGeneratorFromCocoJson(batch_size=8, image_list=images_train, coco=coco_train,
                                             path_folder=os.path.join(DATASET_PATH, 'train'), classes=classes_train, aurgment=True)
    for i in range(1):
        img, mask = train_gen.__getitem__(i)
        pre = model.predict(img)
        gen_viz(img_s=img, mask_s=mask, pred=pre)


def visualizeGenerator(gen, img=None, pred=None):
    import matplotlib as mpl
    colors = ['#0044ff', '#ff00fb', '#ff0000', '#2bff00', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    # Iterate the generator to get image and mask batches
    if gen is not None:
        img1, mask1 = next(gen)
    else:
        img1, mask1 = img, pred
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)

    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)
        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if (i == 1):
                ax.imshow(img1[j])
            else:
                # print(mask.shape)
                for m in range(len(mask1[0, 0, 0, :])):
                    mask_one = mask1[j, :, :, m]
                    if pred is not None:
                        mask_one = mask1[j, :, :, m] > 0.85
                    ax.imshow(mask_one, alpha=0.5)
            ax.axis('off')
            fig.add_subplot(ax)
    plt.show()


# mask color codes
class MaskColorMap(Enum):
    Unlabelled = (0, 0, 0)
    Building = (60, 16, 152)
    Land = (132, 41, 246)
    Road = (110, 193, 228)
    Vegetation = (254, 221, 58)
    Water = (226, 169, 41)

def display_images(instances, rows=2, titles=None):
    """
    :param instances:  list of images
    :param rows: number of rows in subplot
    :param titles: subplot titles
    :return:
    """
    n = len(instances)
    cols = n // rows if (n / rows) % rows == 0 else (n // rows) + 1

    # iterate through images and display subplots
    for j, image in enumerate(instances):
        plt.subplot(rows, cols, j + 1)
        plt.title('') if titles is None else plt.title(titles[j])
        plt.axis("off")
        plt.imshow(image)

    # show the figure
    plt.show()

def rgb_encode_mask(mask):

    # initialize rgb image with equal spatial resolution
    rgb_encode_image = np.zeros((mask.shape[0], mask.shape[1], 3))

    # iterate over MaskColorMap
    for j, cls in enumerate(MaskColorMap):
        # convert single integer channel to RGB channels
        rgb_encode_image[(mask == j)] = np.array(cls.value) / 255.

    # plt.show(rgb_encode_image)

    return rgb_encode_image

def gen_viz(img_s, mask_s, pred = None, epoch = None):
    # print(img_s.shape)
    # print(mask_s.shape)
    fig = plt.figure(figsize=(10, 25))
    fig.suptitle(f'epoch: {str(epoch)}', fontsize=26, fontweight='bold')
    gs = gridspec.GridSpec(nrows=len(img_s), ncols=3)
    colors = ['yellow', 'green', 'red']
    labels = ["Small Bowel", "Large Bowel", "Stomach"]
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3 = mpl.colors.ListedColormap(colors[2])

    mask_s = mask_s[:, :, :, 1:]
    if pred is not None:
        pred = pred[:, :, :, 1:]
    # if pred is not None:
    #     mask_one_hot = tf.keras.utils.to_categorical(pred, num_classes=4)
    #     mask_one_hot = mask_one_hot[:, :, :, 1:]
    #     pred = mask_one_hot

    # if mask_s is not None:
    #     mask_one_hot2 = tf.keras.utils.to_categorical(mask_s, num_classes=4)
    #     mask_one_hot2 = mask_one_hot2[:, :, :, 1:]
    #     mask_s = mask_one_hot2
    if pred is not None:
        print(f'pred shape:{pred.shape}')
    flag = False
    for i in range(0, 8):


        images, mask = img_s[i], mask_s[i]
        sample_img = images
        # print(mask.shape)
        mask1 = mask[:, :, 0]
        mask2 = mask[:, :, 1]
        mask3 = mask[:, :, 2]

        ax0 = fig.add_subplot(gs[i, 0])
        im = ax0.imshow(sample_img)
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

        l0 = ax1.imshow(sample_img)
        l1 = ax1.imshow(np.ma.masked_where(
            mask1 == False, mask1), cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(
            mask2 == False, mask2), cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(
            mask3 == False, mask3), cmap=cmap3, alpha=1)

        if pred is not None:

            pre = pred[i]
            predict1 = pre[:, :, 0] > 0.5
            predict1 = (predict1).astype(np.float32)
            predict1 = np.array(predict1)
            predict2 = pre[:, :, 1] > 0.5
            predict2 = (predict2).astype(np.float32)
            predict2 = np.array(predict2)
            predict3 = pre[:, :, 2] > 0.5
            predict3 = (predict3).astype(np.float32)
            predict3 = np.array(predict3)
            l0 = ax2.imshow(sample_img)
            l1 = ax2.imshow(np.ma.masked_where(
                predict1 == False, predict1), cmap=cmap1, alpha=0.8)
            l2 = ax2.imshow(np.ma.masked_where(
                predict2 == False, predict2), cmap=cmap2, alpha=0.8)
            l3 = ax2.imshow(np.ma.masked_where(
                predict3 == False, predict3), cmap=cmap3, alpha=0.8)
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
