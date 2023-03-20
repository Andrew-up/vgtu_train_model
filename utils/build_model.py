import tensorflow as tf
from keras import backend as K
from keras_unet.models import custom_unet

from utils.model_optimizers import SGD_loss
from utils.model_losses import binary_weighted_cross_entropy


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def unet_model(num_classes: int):
    model = custom_unet(
        input_shape=(128, 128, 3),
        use_batch_norm=True,
        num_classes=num_classes,
        filters=32,
        num_layers=4,
        dropout=0.4,
        activation="relu",
        output_activation='sigmoid'
    )
    iou = MyMeanIOU(num_classes=num_classes)
    loss = binary_weighted_cross_entropy(beta=1.0, is_logits=True)
    model.compile(optimizer=SGD_loss(), loss=loss, metrics=[iou, 'accuracy'])
    return model
