import keras
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))

    def __len__(self):
        # Возвращает количество пакетов, которые генератор будет производить за одну эпоху
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, index):
        # Возвращает один пакет данных для обучения
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = self.x[indexes]
        batch_y = self.y[indexes]
        return batch_x, batch_y


def get_dataset(show_dataset=True, patch_size=300):
    # Загрузка изображения
    img = cv2.imread('img2.png')

    # Размер участков
    # patch_size = 300

    # Размеры карты
    height, width = img.shape[:2]

    img_copy = img.copy()

    # Список для хранения поделенных изображений
    patches = []

    # Список для хранения координат X и Y
    coordinates = []

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.7
    thickness = 2
    color = (0, 0, 255)

    # Перебор строк и столбцов для всех участков на карте
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Определение координат для каждого участка
            x = j // patch_size
            y = i // patch_size

            if i + patch_size > height:
                break
            if j + patch_size > width:
                break

            # Определение границ участка и его вырезание
            patch = img[i:i + patch_size, j:j + patch_size]
            if patch.max() <= 0:
                continue

            # Добавление участка и его координат в соответствующие списки
            patches.append(patch)
            coordinates.append((x, y))

            # Рисование квадрата на исходном изображении и подпись координат на участке
            cv2.rectangle(img_copy, (j, i), (j + patch_size, i + patch_size), (0, 255, 0), 3)
            # cv2.putText(img_copy, f'(x: {x}, y:{y})', (j + 5, i + patch_size // 2), font, font_scale,
            #             color, thickness)


    # Отображение результатов
    if show_dataset:
        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.show()

    print(len(patches), len(coordinates))

    patches = np.array(patches).astype(np.float32)/255
    coordinates = np.array(coordinates).astype(np.uint8)
    return patches, coordinates
