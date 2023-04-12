from pycocotools.coco import COCO
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.io as io
import random
import torchvision.transforms as tf
import pandas as pd
import cv2
from tqdm import tqdm
from copy import deepcopy
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
from torchmetrics import JaccardIndex

from definitions import DATASET_PATH
from utils.unet_pytorch import UNet


def main():
    ann_file_name = 'labels_my-project-name_2022-11-15-02-32-33.json'
    annFile = DATASET_PATH + 'train' + '/' + ann_file_name
    print(annFile)
    train_annotations = COCO(annFile)

    cat_ids = train_annotations.getCatIds(catNms=["Asept", "Bacterial", "Gnoy"])
    train_img_ids = []
    for cat in cat_ids:
        train_img_ids.extend(train_annotations.getImgIds(catIds=cat))

    train_img_ids = list(set(train_img_ids))
    print(f"Number of training images: {len(train_img_ids)}")

    path_dataset = os.path.join(DATASET_PATH, 'train/')

    train_data = ImageData(annotations=train_annotations,
                           img_ids=train_img_ids,
                           cat_ids=cat_ids,
                           root_path=path_dataset,
                           transform=train_transform)
    a = train_data.__getitem__(0)
    return 0

def train_transform(
        img1: torch.Tensor,
        img2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    params = tf.RandomResizedCrop.get_params(img1, scale=(0.5, 1.0), ratio=(0.75, 1.33))

    IMAGE_SIZE = [512, 512]

    gr2rgb = tf.Lambda(lambda x: x.repeat(3, 1, 1))
    img2 = gr2rgb(img2)

    img1 = tf.functional.resized_crop(img1, *params, size=IMAGE_SIZE)
    img2 = tf.functional.resized_crop(img2, *params, size=IMAGE_SIZE)

    return img1, img2


class ImageData(Dataset):
    def __init__(
            self,
            annotations: COCO,
            img_ids: list[int],
            cat_ids: list[int],
            root_path: Path,
            transform
    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [str(root_path + img["file_name"]) for img in self.img_data]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.LongTensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"]
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        m = np.array(mask)
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        if self.transform is not None:
            return self.transform(img, mask)

        return img, mask


if __name__ == "__main__":
    main()
    # viz_model()
