import math
import os.path

from keras.models import Sequential

from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history
from utils.model_callbacks import callback_bce_dice_loss
from controller_vgtu_train.subprocess_train_model_controller import update_model_history
from definitions import MODEL_H5_PATH
from model.model_history import ModelHistory
import tensorflow as tf
from utils.newDataGeneratorCoco import visualizeGenerator


class PrintTrueAndPred(tf.keras.callbacks.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        img, mask = next(iter(self.generator))
        y_true = img
        y_pred = self.model.predict(img)
        print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        visualizeGenerator(img, y_pred, 'DEGUB on_epoch_end')


def train_model(model: Sequential,
                n_epoch,
                path_model,
                batch_size=8,
                dataset_train=None,
                dataset_valid=None,
                dataset_size_train=0,
                dataset_size_val=0,
                model_history: ModelHistory = None,
                monitor='val_my_mean_iou'):
    if model_history:
        model_history.total_epochs = n_epoch
        update_model_history(model_history)

    callback = callback_bce_dice_loss(path=path_model,
                                      monitor=monitor,
                                      mode='max',
                                      model_history=model_history)

    tb_callback = callback.tb_callback()
    reduce_lr = callback.reduce_lr()
    checkpoint = callback.checkpoint()
    print_test = callback.print_test()
    early_stop_train = callback.early_stopping()

    steps_per_epoch = math.ceil(dataset_size_train // batch_size) * 2
    # steps_per_epoch = 15
    validation_steps = math.ceil(dataset_size_val // batch_size)
    # print(f'')
    # validation_steps = len(dataset_valid) // batch_size
    # print(f'steps_per_epoch: {steps_per_epoch}')
    # print(f'validation_steps: {validation_steps}')
    # print('+++++++++++++++++++++++++++++++++++')

    history = model.fit(
        dataset_train,
        validation_data=dataset_valid,
        # validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        # batch_size=1,
        validation_steps=validation_steps,
        callbacks=[tb_callback, reduce_lr, checkpoint, print_test, early_stop_train, PrintTrueAndPred(dataset_train)],
        verbose=True,
        shuffle=False
        # batch_size=batch_size
        # validation_split=0.3
    )
    return history
