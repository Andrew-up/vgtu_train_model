from keras_unet.models import custom_unet
from utils.model_losses import bce_dice_loss, dice_coef, jaccard_distance, iou, jaccard_coef, dice_loss
from utils.model_optimizers import SGD_loss
import tensorflow as tf
from keras import backend as K


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def unet_model(num_classes: int):
    model = custom_unet(
        input_shape=(128, 128, 3),
        use_batch_norm=True,
        num_classes=num_classes,
        filters=16,
        num_layers=4,
        dropout=0.1,
        activation="relu",
        output_activation='sigmoid'
    )
    LR = 1e-4
    opt = tf.keras.optimizers.Adam(LR)
    # model.compile(optimizer=SGD_loss(), loss=bce_dice_loss, metrics=[iou_coef, dice_coef])
    model.compile(optimizer=opt, loss=dice_loss, metrics=[dice_coef, 'accuracy'])
    return model
