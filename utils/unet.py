import keras.losses
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, Conv2DTranspose, concatenate, MaxPooling2D, Dropout
from keras_unet.models import custom_unet
from segmentation_models.losses import cce_dice_loss

from utils.model_optimizers import Adam_opt


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # print(y_true.shape)
        # print(y_pred.shape)
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def conv_block(inputs=None, n_filters=32, dropout_prob=0.0, max_pooling=True):
    conv = Conv2D(n_filters,  # Number of filters
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    if dropout_prob > 0:
        conv = Dropout(rate=dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(2, 2)(conv)

    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(
        n_filters,  # number of filters
        3,  # Kernel size
        strides=2,
        padding='same')(expansive_input)

    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                  3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    return conv

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

def unet_model(input_size=(224, 224, 3), n_filters=32, n_classes=3):
    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same', activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    iou = MyMeanIOU(num_classes=n_classes)
    model.compile(optimizer='adam', loss=cce_dice_loss, metrics=[iou, 'accuracy'])
    return model

def unet(num_classes=None, input_shape=(128, 128, 3)):
    model = custom_unet(
        input_shape=input_shape,
        use_batch_norm=True,
        num_classes=num_classes,
        filters=16,
        num_layers=6,
        # dropout=0.2,
        activation="relu",
        output_activation='softmax',
    )

    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer=Adam_opt(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
