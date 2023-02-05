from matplotlib import gridspec, pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from keras.models import Model, load_model
import tensorflow as tf
from keras.utils import img_to_array

from definitions import ANNOTATION_FILE_PATH, DATASET_PATH, MODEL_H5_PATH
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.get_dataset_coco import filterDataset
from utils.model_losses import dice_coef, bce_dice_loss


def vizualizator(list_images, list_masks):
    x, y = list_images, list_masks
    fig = plt.figure(figsize=(50, 40))
    gs = gridspec.GridSpec(nrows=len(x), ncols=2)
    colors = ['yellow', 'green', 'red']
    labels = ['Asept', 'Bacterial', 'Gnoy']
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3 = mpl.colors.ListedColormap(colors[2])
    flag = False
    for i in range(0, 8):

        images, mask = x[i], y[i]
        sample_img = images
        mask1 = mask[:, :, 0]
        mask2 = mask[:, :, 1]
        mask3 = mask[:, :, 2]

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
        l1 = ax1.imshow(np.ma.masked_where(
            mask1 == False, mask1), cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(
            mask2 == False, mask2), cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(
            mask3 == False, mask3), cmap=cmap3, alpha=1)
        _ = [ax.set_axis_off() for ax in [ax0, ax1]]

        colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]
    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.3,
                        top=0.92, wspace=0.01, hspace=0.08)
    plt.show()

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH)
    model = load_model(MODEL_H5_PATH, custom_objects={'dice_coef': dice_coef,
                                                      'bce_dice_loss': bce_dice_loss})
    for i in range(2):
        train_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                          subset='train',
                                                          input_image_size=(128, 128),
                                                          image_list=images_valid,
                                                          classes=classes,
                                                          coco=coco,
                                                          shuffle=True)

        img_s, mask_s = train_generator_class.__getitem__(1)
        # print(len(img_s))
        print(img_s.shape)
        res = model.predict(img_s)

        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=len(img_s), ncols=4)
        colors = ['yellow', 'green', 'red']
        labels = ['Asept', 'Bacterial', 'Gnoy']
        patches = [mpatches.Patch(
            color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

        cmap1 = mpl.colors.ListedColormap(colors[0])
        cmap2 = mpl.colors.ListedColormap(colors[1])
        cmap3 = mpl.colors.ListedColormap(colors[2])
        flag = False
        for i in range(0, 8):
            list_predict = list()
            list_predict.append(np.sum(res[0, :, :, 0]))
            list_predict.append(np.sum(res[0, :, :, 1]))
            list_predict.append(np.sum(res[0, :, :, 2]))
            max_value = max(list_predict)
            max_index = list_predict.index(max_value)

            images, mask = img_s[i], mask_s[i]
            sample_img = images
            mask1 = mask[:, :, 0]
            mask2 = mask[:, :, 1]
            mask3 = mask[:, :, 2]

            pre = res[i]
            predict1 = pre[:, :, 0]
            predict1 = (predict1 > 0.4).astype(np.float32)
            predict1 = np.array(predict1)

            predict2 = pre[:, :, 1]
            predict2 = (predict2 > 0.4).astype(np.float32)
            predict2 = np.array(predict2)

            predict3 = pre[:, :, 2]
            predict3 = (predict3 > 0.4).astype(np.float32)
            predict3 = np.array(predict3)

            ax0 = fig.add_subplot(gs[i, 0])
            im = ax0.imshow(sample_img)

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
            l1 = ax1.imshow(np.ma.masked_where(
                mask1 == False, mask1), cmap=cmap1, alpha=1)
            l2 = ax1.imshow(np.ma.masked_where(
                mask2 == False, mask2), cmap=cmap2, alpha=1)
            l3 = ax1.imshow(np.ma.masked_where(
                mask3 == False, mask3), cmap=cmap3, alpha=1)

            l0 = ax2.imshow(sample_img)
            l1 = ax2.imshow(np.ma.masked_where(
                predict1 == False, predict1), cmap=cmap1, alpha=0.4)
            l2 = ax2.imshow(np.ma.masked_where(
                predict2 == False, predict2), cmap=cmap2, alpha=0.4)
            l3 = ax2.imshow(np.ma.masked_where(
                predict3 == False, predict3), cmap=cmap3, alpha=0.4)
            _ = [ax.set_axis_off() for ax in [ax0, ax1]]

            colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]

        plt.show()

def viz_model():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH)
    model = load_model(MODEL_H5_PATH, custom_objects={'dice_coef': dice_coef,
                                                      'bce_dice_loss': bce_dice_loss})
    train_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=True)

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
    # test()
    # main()
    # viz_model()
    show_mask_true_and_predict()

