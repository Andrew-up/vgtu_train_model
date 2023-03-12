from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LambdaCallback
import tensorflow as tf
from definitions import TENSORBOARD_LOGS
from model.model_history import ModelHistory


class callback_bce_dice_loss():

    def __init__(self,
                 path=None,
                 monitor='val_dice_coef',
                 mode='auto',
                 model_history_train: ModelHistory() = None):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.model_history: ModelHistory = model_history_train


    def tb_callback(self):
        return tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGS, update_freq=1)


    def checkpoint(self):
        _checkpoint = ModelCheckpoint(
            filepath=self.path,
            monitor=self.monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=self.mode,
        )
        return _checkpoint

    def reduce_lr(self):
        _reduce_lr = ReduceLROnPlateau(monitor=self.monitor,
                                       factor=0.333,
                                       patience=3,
                                       verbose=1,
                                       min_delta=1e-05,
                                       mode=self.mode,
                                       min_lr=1e-04)
        return _reduce_lr

    def early_stopping(self):
        _early_stopping = EarlyStopping(monitor=self.monitor,
                                        min_delta=0.0001,
                                        patience=5,
                                        mode=self.mode)
        return _early_stopping


    def jjjjjjjj(self, epoch, logs):
        if self.model_history:
            print(self.model_history.status)
        # print('22222222222')
        # print(epoch)
        print(logs)
    def print_test(self):
        lambda_callback = LambdaCallback(on_epoch_end=lambda batch, logs: self.jjjjjjjj(batch, logs=logs))
        return lambda_callback

