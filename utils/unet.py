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
from utils.model_losses import dice_loss, dice_coef
from utils.model_optimizers import SGD_loss, Adam_opt

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


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

    iou = MyMeanIOU(num_classes=num_classes)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[iou, dice_coef, 'accuracy'])
    return model
