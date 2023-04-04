import math

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential

from controller_vgtu_train.subprocess_train_model_controller import update_model_history
from model.model_history import ModelHistory
from utils.model_callbacks import callback_bce_dice_loss
from utils.newDataGeneratorCoco import visualizeImageOrGenerator
import numpy as np


class PrintTrueAndPred(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):

        dddddddddd = {
            0: 'treugolnik',
            1: 'kvadrat',
            2: 'krug',
        }

        img, mask = self.generator[0]
        y_true = img
        y_pred = self.model.predict(img).astype(np.float32) > 0.5
        fig, axs = plt.subplots(ncols=4, figsize=(10, 10))
        axs[0].imshow(y_true[0])
        axs[0].set_title('original')
        for i in range(len(y_pred[0, 0, 0, :])):
            axs[i+1].imshow(y_pred[0, :, :, i])
            axs[i+1].set_title(dddddddddd[i])

        plt.show()
        hhhhh = (y_pred).astype(np.uint8)
        print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        # visualizeImageOrGenerator(images_list=y_true, mask_list=hhhhh, subtitle=f'DEGUB on_epoch_end: {epoch}')


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

    callback = callback_bce_dice_loss(path=path_model,
                                      monitor=monitor,
                                      mode=mode,
                                      model_history=model_history)

    tb_callback = callback.tb_callback()
    reduce_lr = callback.reduce_lr()
    checkpoint = callback.checkpoint()
    print_test = callback.print_test()
    early_stop_train = callback.early_stopping()

    steps_per_epoch = math.ceil((dataset_size_train // batch_size) * 20)
    # steps_per_epoch = 10
    # steps_per_epoch = 15
    validation_steps = 2
    # validation_steps = math.ceil(dataset_size_val // batch_size)
    # print(f'')
    # validation_steps = len(dataset_valid) // batch_size
    print(f'steps_per_epoch: {steps_per_epoch}')
    print(f'validation_steps: {validation_steps}')
    # print('+++++++++++++++++++++++++++++++++++')

    history = model.fit(
        dataset_train,
        validation_data=dataset_valid,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        # batch_size=batch_size,
        # validation_steps=validation_steps,
        callbacks=[tb_callback, reduce_lr, checkpoint, print_test, early_stop_train, PrintTrueAndPred(dataset_train)],
        verbose=True,
        shuffle=False
        # batch_size=batch_size
        # validation_split=0.3
    )
    return history
