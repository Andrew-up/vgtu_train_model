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


def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def build_unet(num_classes=None, input_shape=(128, 128, 3)):
    # input layer shape is equal to patch image size
    inputs = Input(shape=input_shape)

    # rescale images from (0, 255) to (0, 1)
    previous_block_activation = inputs  # Set aside residual

    contraction = {}
    # # Contraction path: Blocks 1 through 5 are identical apart from the feature depth
    for f in [16, 32, 64, 128, 256]:
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
        x = Dropout(0.1)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        # x = BatchNormalization()(x)
        contraction[f'conv{f}'] = x
        x = MaxPooling2D((2, 2))(x)
        previous_block_activation = x

    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_block_activation)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(160, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    previous_block_activation = c5

    # Expansive path: Second half of the network: upsampling inputs
    for f in reversed([16, 32, 64, 128, 256]):
        x = Conv2DTranspose(f, (2, 2), strides=(2, 2), padding='same')(previous_block_activation)
        x = concatenate([x, contraction[f'conv{f}']])
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(f, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        previous_block_activation = x

    outputs = Conv2D(filters=num_classes, kernel_size=(1, 1), activation="softmax")(previous_block_activation)
    return Model(inputs=inputs, outputs=outputs)

def unet(num_classes=None, input_shape=(128, 128, 3)):
    model = custom_unet(
        input_shape=input_shape,
        use_batch_norm=True,
        num_classes=num_classes,
        filters=16,
        num_layers=6,
        dropout=0.2,
        activation="relu",
        output_activation='softmax',
    )

    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer=Adam_opt(), loss=cce_dice_loss, metrics=[iou, 'accuracy'])
    return model

