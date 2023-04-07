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

from utils.vizualizators import gen_viz


class PrintTrueAndPred(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        img, mask = self.generator.__getitem__(0)
        # fig, ax = plt.subplots(ncols=3, nrows=4)
        # fig.suptitle(f'epoch: {epoch}', fontsize=20, fontweight='bold')
        y_pred = self.model.predict(img, verbose=1)
        gen_viz(img_s=img, mask_s=mask, pred=y_pred)


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
    steps_per_epoch = int(dataset_size_train // batch_size)

    history = model.fit(
        dataset_train,
        validation_data=dataset_valid,
        validation_steps=3,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        callbacks=[tb_callback, reduce_lr, checkpoint, print_test, PrintTrueAndPred(dataset_train)],
        verbose=True,
        shuffle=False
    )
    return history
