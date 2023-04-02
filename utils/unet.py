# from keras import Input
import keras
import numpy as np
from keras import layers
import tensorflow as tf
from keras import backend as K
from keras.backend import _to_tensor, categorical_crossentropy
from keras_unet.models import custom_unet

from utils.model_optimizers import SGD_loss
from utils.model_losses import binary_weighted_cross_entropy, bce_dice_loss

from utils.loss_functions import Semantic_loss_functions


def dice_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred) - tf.reduce_sum(y_true * y_pred)
    return 1 - numerator / denominator


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


class IOU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=3, name=None, dtype=None):
        super(IOU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        super(IOU, self).update_state(y_true, y_pred, sample_weight)


def jaccard_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=(1, 2))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=(1, 2))
    iou = (intersection + 1e-15) / (sum_ - intersection + 1e-15)
    return 1 - iou


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        x = layers.Dropout(0.3)(x)
        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.Dropout(0.3)(x)

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    model = keras.Model(inputs, outputs)

    iou = MyMeanIOU(num_classes=num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-04), loss=dice_loss, metrics=[iou, 'accuracy'])
    return model
