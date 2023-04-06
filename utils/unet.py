import keras
import tensorflow as tf
from keras import layers, Input

import keras.backend as K
from keras.backend import epsilon


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


class CustomCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name='custom_categorical_crossentropy'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Создаем маску, которая будет устанавливать веса для каждого класса
        mask = tf.cast(tf.math.not_equal(tf.argmax(y_true, axis=-1), 0), tf.float32)
        # Вычисляем потери с учетом маски
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred) * mask
        # Усредняем потери по батчу
        return tf.reduce_mean(loss, axis=-1)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true[..., 1:])  # удаляем первый канал, отвечающий за фон
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def get_model(img_size, num_classes):
    inputs = Input(img_size)
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
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    iou = MyMeanIOU(num_classes=num_classes)

    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[iou, dice_coef, 'accuracy'])
    return model
