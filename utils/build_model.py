import tensorflow as tf
from keras import backend as K
from keras_unet.models import custom_unet

from utils.model_optimizers import SGD_loss
from utils.model_losses import binary_weighted_cross_entropy

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def dice_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred) - tf.reduce_sum(y_true * y_pred)
    return 1 - numerator / denominator


def unet_model(num_classes: int):
    print(num_classes)
    model = custom_unet(
        input_shape=(128, 128, 3),
        use_batch_norm=False,
        num_classes=num_classes,
        filters=32,
        num_layers=4,
        dropout=0.2,
        activation="relu",
        output_activation='softmax'
    )
    iou = MyMeanIOU(num_classes=num_classes)
    # loss = binary_weighted_cross_entropy(beta=0.9, is_logits=True)
    model.compile(optimizer='adam', loss=dice_loss, metrics=[iou, 'accuracy'])
    return model
