from keras.models import Sequential

from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history
from utils.model_callbacks import callback_bce_dice_loss


def train_model(model: Sequential,
                n_epoch,
                path_model,
                batch_size=8,
                dataset_train=None,
                dataset_valid=None,
                dataset_size_train=0):
    model_history = get_last_model_history()
    model_history.total_epochs = n_epoch
    callback = callback_bce_dice_loss(path=path_model,
                                      monitor='val_dice_coef',
                                      mode='max',
                                      model_history=model_history)

    tb_callback = callback.tb_callback()
    reduce_lr = callback.reduce_lr()
    checkpoint = callback.checkpoint()
    print_test = callback.print_test()
    print('dataset_size_train: ' + str(dataset_size_train))
    steps_per_epoch = dataset_size_train // batch_size
    validation_steps = 8

    history = model.fit(
        dataset_train,
        validation_data=dataset_valid,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epoch,
        batch_size=batch_size,
        callbacks=[tb_callback, reduce_lr, checkpoint, print_test],
        verbose=True,
    )
    return history
