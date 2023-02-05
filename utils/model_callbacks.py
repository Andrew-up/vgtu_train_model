from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import tensorflow as tf


class callback_bce_dice_loss():

    def __init__(self, path=None,
                 monitor='val_dice_coef',
                 mode='auto'):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.mode = mode


    def tb_callback(self):
        return tf.keras.callbacks.TensorBoard('./logs', update_freq=1)

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
