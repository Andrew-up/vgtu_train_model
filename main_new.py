import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm

from definitions import DATASET_PATH
from utils.get_dataset_coco import filterDataset
from utils.unet_pytorch import UNet

n_fold = 5
pad_left = 27
pad_right = 27
fine_size = 202
batch_size = 18
epoch = 50
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
        # masks = masks - 1
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            inputs = inputs.float() / 255.
            masks = masks
            masks = masks.squeeze().long()
            logit = model(inputs)
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 1.0, 1.0, 1.0]).to(device))
            loss = criterion(logit, masks)
            loss.backward()
            optimizer.step()

        predicts.append(torch.softmax(logit, dim=0).detach().cpu().numpy())
        truths.append(masks.detach().cpu().numpy())
        running_loss += loss.item() * inputs.size(0)
    # predicts = np.concatenate(predicts).squeeze()
    # truths = np.concatenate(truths).squeeze()



    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()

    jaccard = JaccardIndex(num_classes=4, task="multiclass")

    iou = jaccard(torch.tensor(predicts), torch.tensor(truths))
    epoch_loss = running_loss / data_size
    return epoch_loss, iou


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


def main():
    print(torch.version)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    save_weight = 'weights/'
    if not os.path.isdir(save_weight):
        os.mkdir(save_weight)
    weight_name = 'model_' + str(fine_size + pad_left + pad_right) + '_res18'

    ann_file_name = 'labels_my-project-name_2022-11-15-02-32-33.json'
    # annFile = DATASET_PATH + 'train' + '/' + ann_file_name
    # print(annFile)
    # train_annotations = COCO(annFile)
    train_path = 'train'
    images_train, _, coco_train, classes_train = filterDataset(ann_file_name=ann_file_name,
                                                               percent_valid=0,
                                                               path_folder=train_path,
                                                               shuffie=False
                                                               )
    cat_ids = coco_train.getCatIds(classes_train)

    print(f"Number of training images: {len(images_train)}")

    path_dataset = os.path.join(DATASET_PATH, 'train/')


    train_data = ImageData(annotations=coco_train,
                           image_list=images_train,
                           cat_ids=cat_ids,
                           root_path=path_dataset,
                           transform=True,
                           input_image_size=(128, 128)
                           )

    # return 0
    BATCH_SIZE = 4
    train_dl = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )


    unet = UNet(n_channels=3, n_classes=4)
    unet.to(device)
    scheduler_step = 100
    optimizer = torch.optim.AdamW(unet.parameters())
    # optimizer = torch.optim.SGD(unet.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, scheduler_step, min_lr)

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

        img_list, mask_list = next(iter(train_dl))
        # img_list =
        outputs = unet((img_list.to(device).float() / 255).reshape(4, 3, 128, 128))

        mask_pred = torch.softmax(outputs.squeeze(), dim=0)
        # outputs = outputs.cpu().data.numpy()
        outputs123213 = mask_pred.detach().cpu().numpy()

        # dhdhdhf = np.transpose(outputs[0], (1, 2, 0))
        # dhdhdh123213f = np.transpose(outputs[0], (0, 1, 2))
        display(img_list=img_list,
                mask_list=mask_list,
                mask_pred_list=outputs123213,
                epoch=f'{epoch_ + 1} loss: {round(train_loss, 3)}',
                iou=round(iou.item(), 3))

        print('epoch: {} train_loss: {:.3f} iou: {:.3f}'.format(epoch_ + 1, train_loss, iou))
    return 0


def tensor2numpy(tensor):
    arr = np.array(tensor)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr


def display(img_list, mask_list, mask_pred_list=None, epoch=None, iou=None):
    n_row = img_list.shape[0]
    fig, ax = plt.subplots(ncols=3, nrows=n_row, figsize=(5, 5))
    fig.suptitle(f'epoch: {epoch}.  iou: {iou}', fontsize=16, fontweight='bold')

    ax[0][0].set_title('Фото')
    ax[0][1].set_title('маска')
    ax[0][2].set_title('пред. маска')

    for j in range(n_row):
        ax[j][0].imshow(tensor2numpy(img_list[j]))
        ax[j][0].axis('off')
        ax[j][1].imshow(colorize_mask(tensor2numpy(mask_list[j])))
        ax[j][1].axis('off')
        if mask_pred_list is not None:
            mask_pred = mask_pred_list[j]
            mask_pred[mask_pred < 0.6] = 0
            pred_mask_arg_masx = np.argmax(mask_pred, axis=0)
            # print(f'max: {pred_mask_arg_masx.max()}')
            pred_mask = tensor2numpy(pred_mask_arg_masx)

            ax[j][2].imshow(colorize_mask(pred_mask))
            ax[j][2].axis('off')

    plt.show()


class ImageData(Dataset):
    def __init__(
            self,
            annotations: COCO,
            image_list: list[int],
            cat_ids: list[int],
            root_path: Path,
            transform,
            input_image_size=(128, 128)

    ) -> None:
        super().__init__()
        self.annotations = annotations
        self.img_data = image_list
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
