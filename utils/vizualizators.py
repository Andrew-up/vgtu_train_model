import os.path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from keras.models import Model, load_model

from definitions import ROOT_DIR, DATASET_PATH
from main_new import ImageData, display
from utils.get_dataset_coco import filterDataset
from utils.unet_pytorch import UNet3Plus
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



    device = torch.device('cuda')

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    model = UNet3Plus(num_classes=4)
    x = torch.randn(1, 3, 256, 256)
    path_model = os.path.join(ROOT_DIR, 'weights/model_256_res180.pth')
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.eval()

    torch.onnx.export(model,
                      x,
                      "model_onnx.onnx",
                      export_params=True,
                      verbose=True,
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )

    # onnx_model = onnx.load('model_onnx.onnx')

    onnx2tf.convert(
        input_onnx_file_path="model_onnx.onnx",
        output_folder_path="model.tf",
        output_h5=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
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
                           input_image_size=(256, 256)
                           )

    model = UNet3Plus(num_classes=4)
    path_model = os.path.join(ROOT_DIR, 'weights/model_256_res180.pth')
    model.load_state_dict(torch.load(path_model))
    model.eval()

    for j in range(10):
        img, mask = train_data[random.randrange(1, 400)]
        img2 = np.transpose(img.numpy(), (1, 2, 0))
        res = img[None, :, :, :]
        res11 = model(res.float())
        img_out = torch.softmax(res11.squeeze(), dim=0)
        mask_res = np.argmax(img_out.detach().numpy(), axis=0)
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
    print(output_data.shape)
    plt.imshow(output_data)
    plt.show()

if __name__ == "__main__":
    # torch_to_onnx()
    # pass
    test_tensorflow_model()

    # load_model()
# pppppppppp()
# main()
# viz_model()
# show_mask_true_and_predict()
