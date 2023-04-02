import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, Conv2DTranspose, Concatenate, \
    BatchNormalization
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import MeanIoU
import keras.backend as K
import keras


from utils.unet_new import MyMeanIOU, dice_loss

def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))
    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return tf.reduce_mean(jd)


def unet_gpt(input_shape, num_classes):
    inputs = Input(input_shape)
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Decoder
    up5 = Conv2DTranspose(256, 3, strides=(2, 2), padding='same')(drop4)
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same')(conv5)
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same')(conv6)
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output
    output = Conv2D(3, 1, activation='softmax', name='output')(conv7)
    # Создание модели

    model = Model(inputs=inputs, outputs=output)
    iou = MyMeanIOU(num_classes=num_classes)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=jaccard_distance,
                  metrics=[iou])

    return model