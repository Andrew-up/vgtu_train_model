import os.path

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
from utils.model_losses import dice_coef, bce_dice_loss, jaccard_distance, iou, jaccard_coef, dice_loss


def vizualizator(list_images, list_masks):
    x, y = list_images, list_masks
    fig = plt.figure(figsize=(50, 40))
    gs = gridspec.GridSpec(nrows=len(x), ncols=2)
    colors = ['#705335', '#25221B', '#E63244', '#EC7C26', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
              '#6C4675', '#969992', '#9E9764']
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'треугольник', 'квадрат', 'круг']
    patches = [mpatches.Patch(
        color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3 = mpl.colors.ListedColormap(colors[2])
    cmap4 = mpl.colors.ListedColormap(colors[3])
    cmap5 = mpl.colors.ListedColormap(colors[4])
    cmap6 = mpl.colors.ListedColormap(colors[5])
    cmap7 = mpl.colors.ListedColormap(colors[6])
    cmap8 = mpl.colors.ListedColormap(colors[7])
    cmap9 = mpl.colors.ListedColormap(colors[8])
    cmap10 = mpl.colors.ListedColormap(colors[9])
    cmap11 = mpl.colors.ListedColormap(colors[10])
    cmap12 = mpl.colors.ListedColormap(colors[11])
    flag = False
    for i in range(0, 2):

        images, mask = x[i], y[i]
        sample_img = images
        mask1 = mask[:, :, 0]
        mask2 = mask[:, :, 1]
        mask3 = mask[:, :, 2]
        mask4 = mask[:, :, 3]
        mask5 = mask[:, :, 4]
        mask6 = mask[:, :, 5]
        mask7 = mask[:, :, 6]
        mask8 = mask[:, :, 7]
        mask9 = mask[:, :, 8]
        mask10 = mask[:, :, 9]
        mask11 = mask[:, :, 10]
        mask12 = mask[:, :, 11]

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
        l4 = ax1.imshow(np.ma.masked_where(
            mask4 == False, mask4), cmap=cmap4, alpha=1)
        l5 = ax1.imshow(np.ma.masked_where(
            mask5 == False, mask5), cmap=cmap5, alpha=1)
        l6 = ax1.imshow(np.ma.masked_where(
            mask6 == False, mask6), cmap=cmap6, alpha=1)
        l7 = ax1.imshow(np.ma.masked_where(
            mask7 == False, mask7), cmap=cmap7, alpha=1)
        l8 = ax1.imshow(np.ma.masked_where(
            mask8 == False, mask8), cmap=cmap8, alpha=1)
        l9 = ax1.imshow(np.ma.masked_where(
            mask9 == False, mask9), cmap=cmap9, alpha=1)
        l10 = ax1.imshow(np.ma.masked_where(
            mask10 == False, mask10), cmap=cmap10, alpha=1)
        l11 = ax1.imshow(np.ma.masked_where(
            mask11 == False, mask11), cmap=cmap11, alpha=1)
        l12 = ax1.imshow(np.ma.masked_where(
            mask12 == False, mask12), cmap=cmap12, alpha=1)
        _ = [ax.set_axis_off() for ax in [ax0, ax1]]

        #
        colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]]
    plt.subplots_adjust(left=0.11, bottom=0.08, right=0.3,
                        top=0.92, wspace=0.01, hspace=0.08)
    plt.show()


from keras import backend as K


def show_mask_true_and_predict():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=0)
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_10.h5')
    # model = load_model(paths_m)
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss,
                                                # 'jaccard_distance': jaccard_distance,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })
    # model = load_model(model)
    for i in range(1):
        train_generator_class = DataGeneratorFromCocoJson(batch_size=4,
                                                          subset='train',
                                                          input_image_size=(128, 128),
                                                          image_list=images_train,
                                                          classes=classes,
                                                          coco=coco,
                                                          shuffle=False)

        img_s, mask_s = train_generator_class.__getitem__(0)
        # print(len(img_s))
        # plt.imshow(img_s[1, :, :, :])
        # plt.show()
        # print(img_s.shape)
        res = model.predict(img_s)

        fig = plt.figure(figsize=(10, 25))
        gs = gridspec.GridSpec(nrows=len(img_s), ncols=4)
        colors = ['#705335', '#25221B', '#E63244', '#EC7C26', '#474B4E', '#D84B20', '#8F8F8F', '#6D6552', '#4E5754',
                  '#6C4675', '#969992',
                  '#9E9764']
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'треугольник', 'квадрат', 'круг']
        patches = [mpatches.Patch(
            color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

        cmap1 = mpl.colors.ListedColormap(colors[0])
        cmap2 = mpl.colors.ListedColormap(colors[1])
        cmap3 = mpl.colors.ListedColormap(colors[2])
        cmap4 = mpl.colors.ListedColormap(colors[3])
        cmap5 = mpl.colors.ListedColormap(colors[4])
        cmap6 = mpl.colors.ListedColormap(colors[5])
        cmap7 = mpl.colors.ListedColormap(colors[6])
        cmap8 = mpl.colors.ListedColormap(colors[7])
        cmap9 = mpl.colors.ListedColormap(colors[8])
        cmap10 = mpl.colors.ListedColormap(colors[9])
        cmap11 = mpl.colors.ListedColormap(colors[10])
        cmap12 = mpl.colors.ListedColormap(colors[11])

        flag = False
        for i in range(0, 4):
            # list_predict = list()
            # list_predict.append(np.sum(res[0, :, :, 0]))
            # list_predict.append(np.sum(res[0, :, :, 1]))
            # list_predict.append(np.sum(res[0, :, :, 2]))
            # max_value = max(list_predict)
            # max_index = list_predict.index(max_value)

            images, mask = img_s[i], mask_s[i]
            sample_img = images
            # print(sample_img.shape)
            # print(type(sample_img))
            # print()
            mask1 = mask[:, :, 0]
            mask2 = mask[:, :, 1]
            mask3 = mask[:, :, 2]
            mask4 = mask[:, :, 3]
            mask5 = mask[:, :, 4]
            mask6 = mask[:, :, 5]
            mask7 = mask[:, :, 6]
            mask8 = mask[:, :, 7]
            mask9 = mask[:, :, 8]
            mask10 = mask[:, :, 9]
            mask11 = mask[:, :, 10]
            mask12 = mask[:, :, 11]

            pre = res[i]
            predict1 = pre[:, :, 0]
            predict1 = (predict1 > 0.9).astype(np.float32)
            predict1 = np.array(predict1)

            predict2 = pre[:, :, 1]
            predict2 = (predict2 > 0.9).astype(np.float32)
            predict2 = np.array(predict2)

            predict3 = pre[:, :, 2]
            predict3 = (predict3 > 0.9).astype(np.float32)
            predict3 = np.array(predict3)

            predict4 = pre[:, :, 3]
            predict4 = (predict4 > 0.9).astype(np.float32)
            predict4 = np.array(predict4)

            predict5 = pre[:, :, 4]
            predict5 = (predict5 > 0.9).astype(np.float32)
            predict5 = np.array(predict5)

            predict6 = pre[:, :, 5]
            predict6 = (predict6 > 0.9).astype(np.float32)
            predict6 = np.array(predict6)

            predict7 = pre[:, :, 6]
            predict7 = (predict7 > 0.9).astype(np.float32)
            predict7 = np.array(predict7)

            predict8 = pre[:, :, 7]
            predict8 = (predict8 > 0.9).astype(np.float32)
            predict8 = np.array(predict8)

            predict9 = pre[:, :, 8]
            predict9 = (predict9 > 0.9).astype(np.float32)
            predict9 = np.array(predict9)

            predict10 = pre[:, :, 9]
            predict10 = (predict10 > 0.9).astype(np.float32)
            predict10 = np.array(predict10)

            predict11 = pre[:, :, 10]
            predict11 = (predict11 > 0.9).astype(np.float32)
            predict11 = np.array(predict11)

            predict12 = pre[:, :, 11]
            predict12 = (predict12 > 0.9).astype(np.float32)
            predict12 = np.array(predict12)

            a = pre[0, 0, :]
            # for isssssssssss in pre[]:
            #     print(isssssssssss)
            #     jsssdf =

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
            l1 = ax1.imshow(np.ma.masked_where(
                mask1 == False, mask1), cmap=cmap1, alpha=1)
            l2 = ax1.imshow(np.ma.masked_where(
                mask2 == False, mask2), cmap=cmap2, alpha=1)
            l3 = ax1.imshow(np.ma.masked_where(
                mask3 == False, mask3), cmap=cmap3, alpha=1)
            l4 = ax1.imshow(np.ma.masked_where(
                mask4 == False, mask4), cmap=cmap4, alpha=1)
            l5 = ax1.imshow(np.ma.masked_where(
                mask5 == False, mask5), cmap=cmap5, alpha=1)
            l6 = ax1.imshow(np.ma.masked_where(
                mask6 == False, mask6), cmap=cmap6, alpha=1)
            l7 = ax1.imshow(np.ma.masked_where(
                mask7 == False, mask7), cmap=cmap7, alpha=1)
            l8 = ax1.imshow(np.ma.masked_where(
                mask8 == False, mask8), cmap=cmap8, alpha=1)
            l9 = ax1.imshow(np.ma.masked_where(
                mask9 == False, mask9), cmap=cmap9, alpha=1)
            l10 = ax1.imshow(np.ma.masked_where(
                mask10 == False, mask10), cmap=cmap10, alpha=1)
            l11 = ax1.imshow(np.ma.masked_where(
                mask11 == False, mask11), cmap=cmap11, alpha=1)
            l12 = ax1.imshow(np.ma.masked_where(
                mask12 == False, mask12), cmap=cmap12, alpha=1)
            _ = [ax.set_axis_off() for ax in [ax0, ax1]]

            l0 = ax2.imshow(sample_img)
            # l1 = ax2.imshow(np.ma.masked_where(
            #     predict1 == False, predict1), cmap=cmap1, alpha=0.4)
            # l2 = ax2.imshow(np.ma.masked_where(
            #     predict2 == False, predict2), cmap=cmap2, alpha=0.4)
            # l3 = ax2.imshow(np.ma.masked_where(
            #     predict3 == False, predict3), cmap=cmap3, alpha=0.4)
            # l4 = ax2.imshow(np.ma.masked_where(
            #     predict4 == False, predict4), cmap=cmap4, alpha=0.4)
            # l5 = ax2.imshow(np.ma.masked_where(
            #     predict5 == False, predict5), cmap=cmap5, alpha=0.4)
            # l6 = ax2.imshow(np.ma.masked_where(
            #     predict6 == False, predict6), cmap=cmap6, alpha=0.4)
            # l7 = ax2.imshow(np.ma.masked_where(
            #     predict7 == False, predict7), cmap=cmap7, alpha=0.4)
            # l8 = ax2.imshow(np.ma.masked_where(
            #     predict8 == False, predict8), cmap=cmap8, alpha=0.4)
            # l9 = ax2.imshow(np.ma.masked_where(
            #     predict9 == False, predict9), cmap=cmap9, alpha=0.4)
            l10 = ax2.imshow(np.ma.masked_where(
                predict10 == False, predict10), cmap=cmap10, alpha=1)
            l11 = ax2.imshow(np.ma.masked_where(
                predict11 == False, predict11), cmap=cmap11, alpha=0.8)
            l12 = ax2.imshow(np.ma.masked_where(
                predict12 == False, predict12), cmap=cmap12, alpha=0.5)
            _ = [ax.set_axis_off() for ax in [ax0, ax1]]

            colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]]

        plt.show()


def pppppppppp():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=0)
    paths_m = os.path.join(MODEL_H5_PATH, 'model_1_0_10.h5')
    # model = load_model(paths_m)
    model = load_model(paths_m, custom_objects={'dice_loss': dice_loss,
                                                # 'jaccard_distance': jaccard_distance,
                                                # 'dice_coef': dice_coef,
                                                # 'jaccard_coef': jaccard_coef
                                                })
    train_generator_class = DataGeneratorFromCocoJson(batch_size=4,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)

    img_s, mask_s = train_generator_class.__getitem__(0)

    def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = tf.expand_dims(pred_mask, axis=-1)
        return pred_mask

    def display_sample(display_list):
        plt.figure(figsize=(16, 16))
        title = ['Input Image', 'True Mask', 'Predicted Mask']
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()
    def show_predictions():
        one_img = img_s[0, :, :, :][tf.newaxis, ...]
        print(one_img.shape)
        print('----------------------111111111111')
        prediction = model.predict(one_img)
        pred_mask = create_mask(prediction)
        print('----------------------33333333333333')

        display_sample([img_s[0, :, :, :], mask_s[0, :, :, :]])

    show_predictions()



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
#     # test()
    pppppppppp()
#     # main()
#     # viz_model()
#     show_mask_true_and_predict()
