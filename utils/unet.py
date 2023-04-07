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

from utils.model_optimizers import SGD_loss, Adam_opt


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def dice_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred) - tf.reduce_sum(y_true * y_pred)
    return 1 - numerator / denominator

def unet(num_classes=None, input_shape=(128, 128, 3)):

    model = custom_unet(
        input_shape=input_shape,
        use_batch_norm=True,
        num_classes=num_classes,
        filters=32,
        num_layers=5,
        dropout=0.3,
        activation="relu",
        output_activation='softmax'
    )
    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer=Adam_opt(), loss=dice_loss, metrics=[iou_coef, iou, 'accuracy'])
    return model