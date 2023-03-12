import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, load_model
from keras.utils import img_to_array
from utils.vizualizators import vizualizator
from definitions import MODEL_H5_PATH, ANNOTATION_FILE_PATH, DATASET_PATH
from utils.DataGeneratorFromCocoJson import DataGeneratorFromCocoJson
from utils.build_model import unet_model
from utils.get_dataset_coco import filterDataset
from utils.model_losses import dice_coef, bce_dice_loss
from utils.model_train import train_model


def main():
    images_train, images_valid, coco, classes = filterDataset(ANNOTATION_FILE_PATH, percent_valid=30)


    # return 0
    train_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_train,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)

    valid_generator_class = DataGeneratorFromCocoJson(batch_size=8,
                                                      subset='train',
                                                      input_image_size=(128, 128),
                                                      image_list=images_valid,
                                                      classes=classes,
                                                      coco=coco,
                                                      shuffle=False)


    img_list, img_mask = train_generator_class.__getitem__(1)
    vizualizator(img_list, img_mask)
    # print(img_mask)

    model = unet_model(MODEL_H5_PATH)
    # print(model.summary())
    # return 0


    history = train_model(path_model=MODEL_H5_PATH,
                          model=model,
                          n_epoch=200,
                          batch_size=8,
                          dataset_train=train_generator_class,
                          dataset_valid=valid_generator_class,
                          dataset_size_train=len(images_train))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for IOU
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('dice_coef iou')
    plt.ylabel('iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


    model.save(MODEL_H5_PATH)


if __name__ == "__main__":
    main()
    # viz_model()
