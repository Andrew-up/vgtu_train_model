import math

import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend import epsilon
from keras.models import Sequential

from controller_vgtu_train.subprocess_train_model_controller import update_model_history
from model.model_history import ModelHistory
from utils.model_callbacks import callback_function
import keras.backend as K

colors = [
    [0, 0, 0],   # Красный
    [0, 255, 0],   # Зеленый
    [0, 0, 255],   # Синий
    [255, 255, 0]  # Желтый
]
def color_mask(mask):
    mask = np.argmax(mask, axis=-1)
    mask = mask[:, :, np.newaxis]
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

class PrintTrueAndPred(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        img, mask = self.generator[0]
        fig, ax = plt.subplots(ncols=3, nrows=4)
        fig.suptitle(f'epoch: {epoch}', fontsize=20, fontweight='bold')
        y_pred = self.model.predict(img, verbose=1)
        ax[0][0].title.set_text('original image')
        ax[0][1].title.set_text('original mask')
        ax[0][2].title.set_text('predict mask image')
        for i in range(4):
            # y_pred3 = np.argmax(y_pred[i], axis=-1)
            ax[i][0].imshow(img[i])
            ax[i][1].imshow(color_mask(mask[i]))
            mask123123 = np.argmax(y_pred[i], axis=-1)
            mask1231 = tf.keras.utils.to_categorical(mask123123)
            ax[i][2].imshow(color_mask(mask1231))
            ax[i][0].axis('off')
            ax[i][1].axis('off')
            ax[i][2].axis('off')

        plt.show()


def train_model(model: Sequential,
                n_epoch,
                path_model,
                batch_size=8,
                dataset_train=None,
                dataset_valid=None,
                dataset_size_train=0,
                dataset_size_val=0,
                model_history: ModelHistory = None,
                monitor='my_mean_iou',
                mode='max'):
    if model_history:
        model_history.total_epochs = n_epoch
        update_model_history(model_history)

    callback = callback_function(path=path_model,
                                 monitor=monitor,
                                 mode=mode,
                                 model_history=model_history)

    tb_callback = callback.tb_callback()
    reduce_lr = callback.reduce_lr()
    checkpoint = callback.checkpoint()
    print_test = callback.print_test()
    early_stop_train = callback.early_stopping()

    steps_per_epoch = math.ceil((dataset_size_train // batch_size) * 50)

    history = model.fit(
        dataset_train,
        validation_data=dataset_valid,
        validation_steps=10,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        callbacks=[tb_callback, reduce_lr, checkpoint, print_test, PrintTrueAndPred(dataset_train)],
        verbose=True,
        shuffle=False
    )
    return history
