import os.path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from keras.models import Model, load_model

from definitions import ROOT_DIR, DATASET_PATH
from main_new import ImageData, display
from utils.get_dataset_coco import filterDataset
from utils.unet_pytorch import UNet
import onnx
import onnx2tf



def viz_torch(img, mask):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()


def torch_to_onnx(batch_size=4):
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    model = UNet(n_classes=4, n_channels=3)
    # model = UNet3Plus(num_classes=4)
    x = torch.randn(1, 3, 32, 32)
    path_model = os.path.join(ROOT_DIR, 'weights/Unet_model-0.356.pth')
    onnx_path = os.path.join(ROOT_DIR, 'weights/model_onnx.onnx')
    res_path = os.path.join(ROOT_DIR, 'weights/result_model')
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.eval()

    # print(model)
    # return 0

    torch.onnx.export(model,
                      x,
                      onnx_path,
                      export_params=True,
                      verbose=True,
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )

    # onnx_model = onnx.load('model_onnx.onnx')
    # print(type(onnx_model))

    # print(onnx_path)
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=res_path,
        # output_h5=True,
        # copy_onnx_input_output_names_to_tflite=True,
        # output_nms_with_dynamic_tensor=True,
        # replace_argmax_to_reducemax_and_indicies_is_float32=True,
        # non_verbose=True
    )




    # test123 = convert(input_onnx_file_path='model_onnx.onnx',
    #                   output_h5=True,
    #                   output_folder_path=ROOT_DIR)

    # onnx_model = onnx.load('model_onnx.onnx')
    # tf_rep = prepare(onnx_model)

    print()
    # onnx_model = onnx.load("model_onnx.onnx")  # load onnx model
    # tf_rep = prepare(onnx_model)  # prepare tf representation
    # tf_rep.export_graph("tf_mode.h5")  # export the model


def load_model222():
    ann_file_name = 'labels_my-project-name_2022-11-15-02-32-33.json'
    # annFile = DATASET_PATH + 'train' + '/' + ann_file_name
    # print(annFile)
    # train_annotations = COCO(annFile)
    train_path = 'train/'

    images_train, _, coco_train, classes_train = filterDataset(ann_file_name=ann_file_name,
                                                               percent_valid=0,
                                                               path_folder=train_path,
                                                               shuffie=False
                                                               )
    cat_ids = coco_train.getCatIds(classes_train)
    path_dataset = os.path.join(DATASET_PATH, train_path)

    train_data = ImageData(annotations=coco_train,
                           image_list=images_train,
                           cat_ids=cat_ids,
                           root_path=path_dataset,
                           transform=True,
                           input_image_size=(128, 128)
                           )

    model = UNet(n_classes=4, n_channels=3)
    path_model = os.path.join(ROOT_DIR, 'weights/Unet_model-0.408.pth')
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.eval()


    for j in range(10):
        img, mask = train_data[random.randrange(1, 400)]
        img2 = np.transpose(img.numpy(), (1, 2, 0))
        res = img[None, :, :, :]
        res11 = model(res.float())
        img_out = torch.softmax(res11.squeeze(), dim=0)

        sghsfdhs = img_out.detach().numpy()
        mask_res = np.argmax(img_out.detach().numpy(), axis=0)
        # hghjfdgfd = mask_res.numpy()
        display(img2, mask, mask_res)



    # outputs = outputs['final_pred']
    # img_out = torch.softmax(outputs.squeeze(), dim=0)
    # print()


import tensorflow as tf
from PIL import Image
def test_tensorflow_model():
    interpreter = tf.lite.Interpreter(model_path='model.tf/model_onnx_float32.tflite')
    interpreter.allocate_tensors()
    image_path = '../dataset/train/0108.png'
    image = Image.open(image_path)

    # Изменение размера изображения
    new_size = (256, 256)
    image = image.resize(new_size)
    image_tensor = tf.keras.preprocessing.image.img_to_array(image)
    image_tensor = tf.expand_dims(image_tensor, axis=0)


    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print(input_shape)
    input_data = np.array(image_tensor, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = tf.nn.softmax(output_data, axis=-1)
    output_data = np.squeeze(output_data)
    output_data = np.argmax(output_data, axis=-1)
    display(image, output_data)



colors = [
    [127, 127, 127],  # фон
    [0, 255, 0],  # Зеленый
    [0, 0, 255],  # Синий
    [255, 255, 0]  # Желтый
]

def colorize_mask(mask):
    # Определяем количество классов и создаем пустой массив для цветовых масок
    num_classes = np.max(mask) + 1
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Создаем цветовую маску для каждого класса
    for i in range(num_classes):
        # Используем заданный цвет для каждого класса
        color = colors[i]
        # Применяем маску и цвет для каждого класса
        color_mask[mask == i] = color

    return color_mask




if __name__ == "__main__":
    torch_to_onnx()
    # pass
    # test_tensorflow_model()

    # load_model222()
# pppppppppp()
# main()
# viz_model()
# show_mask_true_and_predict()
