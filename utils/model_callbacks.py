import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LambdaCallback

from controller_vgtu_train.subprocess_train_model_controller import update_model_history
from definitions import TENSORBOARD_LOGS
from model.model_history import ModelHistory


class callback_bce_dice_loss():

    def __init__(self,
                 path=None,
                 monitor='val_dice_coef',
                 mode='auto',
                 model_history: ModelHistory = None):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.result_model_history: ModelHistory() = model_history

    def tb_callback(self):
        return tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGS, update_freq=1)

    def checkpoint(self):
        _checkpoint = ModelCheckpoint(
            filepath=self.path,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=self.mode,
            monitor=self.monitor
        )
        return _checkpoint

    def reduce_lr(self):
        _reduce_lr = ReduceLROnPlateau(monitor=self.monitor,
                                       factor=0.05,
                                       patience=3,
                                       verbose=1,
                                       min_delta=1e-05,
                                       mode=self.mode,
                                       min_lr=1e-08)
        return _reduce_lr

    def early_stopping(self):
        _early_stopping = EarlyStopping(monitor=self.monitor,
                                        min_delta=0.0001,
                                        patience=5,
                                        mode=self.mode)
        return _early_stopping

    def on_epoch_end_update(self, epoch, logs):
        if self.result_model_history:
            # print(type(epoch))
            # print(epoch)
            # print(logs)
            self.result_model_history.current_epochs = epoch
            # update_model_history(self.result_model_history)

    def on_train_end_update(self, logs):
        if self.result_model_history:
            print("Завершено обучение модели")

    def on_train_begin_update(self, logs):
        if self.result_model_history:
            self.result_model_history.status = 'train'
            update_model_history(self.result_model_history)

    def print_test(self):
        lambda_callback = LambdaCallback(on_epoch_end=lambda batch, logs: self.on_epoch_end_update(batch, logs=logs),
                                         on_train_end=lambda logs: self.on_train_end_update(logs),
                                         on_train_begin=lambda logs: self.on_train_begin_update(logs))
        return lambda_callback
