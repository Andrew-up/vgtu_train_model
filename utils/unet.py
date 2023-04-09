import keras
import tensorflow as tf
from keras import layers, Input, Model

import keras.backend as K
from keras.backend import epsilon
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, \
    Lambda, Concatenate
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras_unet.models import custom_unet

from utils.loss_functions import Semantic_loss_functions
from utils.model_losses import dice_loss_fun
from utils.model_optimizers import SGD_loss, Adam_opt


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

def unet(num_classes=None, input_shape=(128, 128, 3)):
    model = custom_unet(
        input_shape=input_shape,
        use_batch_norm=True,
        num_classes=num_classes,
        filters=32,
        num_layers=4,
        dropout=0.3,
        activation="relu",
        output_activation='softmax'
    )
    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer=SGD_loss(), loss=DiceLoss(), metrics=[iou, 'accuracy'])
    return model
