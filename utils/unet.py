from typing import Union, Callable

import keras
import numpy as np
import tensorflow as tf
from keras import layers, Input, Model

import keras.backend as K
from keras.backend import epsilon
from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, \
    Lambda, Concatenate, UpSampling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras_unet.models import custom_unet
from segmentation_models.losses import dice_loss, jaccard_loss, cce_dice_loss

from utils.loss_functions import Semantic_loss_functions
# from utils.model_losses import dice_loss, dice_coef
from utils.model_optimizers import SGD_loss, Adam_opt

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # print(y_true.shape)
        # print(y_pred.shape)
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def dice_loss_new(y_true, y_pred):

    smooth = 1e-7
    dice = 0
    num_classes = 3
    for i in range(num_classes):
        y_true_i = y_true[:, :, :, i]
        y_pred_i = y_pred[:, :, :, i]
        intersection = K.sum(y_true_i * y_pred_i)
        union = K.sum(y_true_i) + K.sum(y_pred_i)
        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice += dice_i

    return 1 - dice / num_classes

def unet(num_classes=None, input_shape=(128, 128, 3)):
    model = custom_unet(
        input_shape=input_shape,
        use_batch_norm=False,
        num_classes=num_classes,
        filters=16,
        num_layers=6,
        dropout=0.2,
        activation="relu",
        output_activation='softmax',
    )

    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer=Adam_opt(), loss=dice_loss_new, metrics=[iou, 'accuracy'])
    return model

