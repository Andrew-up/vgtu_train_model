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

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_loss = - alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
        return K.mean(focal_loss, axis=-1)
    return focal_loss_fixed
class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


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
        num_layers=4,
        dropout=0.2,
        activation="relu",
        output_activation='softmax'
    )
    iou = MyMeanIOU(num_classes=num_classes, ignore_class=0)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[iou, 'accuracy'])
    return model
