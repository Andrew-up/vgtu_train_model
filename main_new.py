import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
from definitions import DATASET_PATH
from utils.unet_pytorch import UNet
from utils.vizualizators import viz_torch

n_fold = 5
pad_left = 27
pad_right = 27
fine_size = 202
batch_size = 18
epoch = 10
snapshot = 6
max_lr = 0.012
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
# n_fold = 5
device = torch.device('cuda')


def train(train_loader, model, optimizer, data_size):
      running_loss = 0.0
      # data_size = len(train_data)

      model.train()
      predicts = []
      truths = []

      for inputs, masks in tqdm(train_loader):
        #print(inputs.shape)
        #print(masks.shape)
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
          inputs = inputs.float()
          masks = masks
          masks = masks.squeeze()
          logit = model(inputs)
          criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 1.0, 1, 1]).to(device))
          loss = criterion(logit, masks)
          loss.backward()
          optimizer.step()

        predicts.append(torch.softmax(logit, dim=0).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())

        #predicts = np.concatenate(predicts).squeeze()
        #truths = np.concatenate(truths).squeeze()

        running_loss += loss.item() * inputs.size(0)

      predicts = np.concatenate(predicts).squeeze()
      truths = np.concatenate(truths).squeeze()

      jaccard = JaccardIndex(num_classes=4, task="multiclass")

      iou = jaccard(torch.tensor(predicts), torch.tensor(truths))
      epoch_loss = running_loss / data_size
      return epoch_loss, iou


def main():
    print(torch.version)
    print(torch.cuda.is_available())

    print(torch.version.cuda)

    save_weight = 'weights/'
    if not os.path.isdir(save_weight):
        os.mkdir(save_weight)
    weight_name = 'model_' + str(fine_size + pad_left + pad_right) + '_res18'

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
                           transform=True,
                           input_image_size=(128, 128)
                           )
    img, mask = train_data[111]
    # print(train_data.__len__())
    print(mask.shape)
    viz_torch(img, mask)
    unet = UNet(n_channels=3, n_classes=3)
    print(unet)
    unet.to(device)
    scheduler_step = 100
    optimizer = torch.optim.AdamW(unet.parameters())
    # optimizer = torch.optim.SGD(unet.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)

    BATCH_SIZE = 4

    train_dl = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    best_param = None
    data_size = int(train_data.__len__())
    num_snapshot = 0
    best_iou = 0

    for epoch_ in range(epoch):
        train_loss, iou = train(train_dl, unet, optimizer, data_size=data_size)
        lr_scheduler.step()

        if iou > best_iou:
            best_iou = iou
            best_param = unet.state_dict()

        if epoch_ == epoch - 1:
            torch.save(best_param, save_weight + weight_name + str(num_snapshot) + '.pth')



        print('epoch: {} train_loss: {:.3f} iou: {:.3f}'.format(epoch_ + 1, train_loss, iou))

    return 0


class ImageData(Dataset):
    def __init__(
            self,
            annotations: COCO,
            img_ids: list[int],
            cat_ids: list[int],
            root_path: Path,
            transform,
            input_image_size=(128, 128)

    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [str(root_path + img["file_name"]) for img in self.img_data]
        self.transform = transform
        self.input_image_size = input_image_size

    def __len__(self) -> int:
        return len(self.files)

    def train_transform(self,
                        img1: torch.Tensor,
                        img2: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        params = tf.RandomResizedCrop.get_params(img1, scale=[0.5, 1.0], ratio=[0.75, 1.33])
        img1 = tf.functional.resized_crop(img1, *params, size=self.input_image_size, antialias=True)
        img2 = tf.functional.resized_crop(img2, *params, size=self.input_image_size, antialias=True)

        return img1, img2

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )
        anns = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"]
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        img = io.read_image(self.files[i])

        img = torchvision.transforms.Resize(self.input_image_size, antialias=True)(img)
        mask = torchvision.transforms.Resize(self.input_image_size, antialias=True)(mask)


        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        if self.transform:
            return self.train_transform(img1=img, img2=mask)

        return img, mask


if __name__ == "__main__":
    main()
