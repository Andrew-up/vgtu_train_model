from keras_unet.models import custom_unet
from utils.model_losses import bce_dice_loss, dice_coef
from utils.model_optimizers import SGD_loss

def unet_model(path_model):
    model = custom_unet(
        input_shape=(128, 128, 3),
        use_batch_norm=True,
        num_classes=3,
        filters=64,
        num_layers=4,
        dropout=0.1,
        activation="relu",
        output_activation='sigmoid'
    )

    model.compile(optimizer=SGD_loss(), loss=bce_dice_loss, metrics=[dice_coef])

    # print(model.summary())
    return model
