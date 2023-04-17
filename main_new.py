import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.io as io
import torchvision.transforms as tf
import torchvision.transforms.functional as functional
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm

from definitions import DATASET_PATH
from utils.get_dataset_coco import filterDataset
from utils.unet_pytorch import UNet
import torchvision.transforms as T
import torch.nn.functional as F

n_fold = 5
pad_left = 27
pad_right = 27
fine_size = 202
# batch_size = 18
epoch = 50
snapshot = 6
max_lr = 0.012
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4
# n_fold = 5
device = torch.device('cuda')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=3):
    # print(f'pred_mask shpe: {pred_mask.shape}')
    # print(f'mask shpe: {mask.shape}')
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)


        # test123 = mask.detach().cpu().numpy()
        # test123111 = pred_mask.detach().cpu().numpy()

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, patch=False):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            # if patch:
            #     bs, n_tiles, c, h, w = image_tiles.size()
            #
            #     image_tiles = image_tiles.view(-1, c, h, w)
            #     mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.float().to(device)
            mask = mask_tiles.squeeze().to(device)

            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient
            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    # if patch:
                    #     bs, n_tiles, c, h, w = image_tiles.size()
                    #
                    #     image_tiles = image_tiles.view(-1, c, h, w)
                    #     mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.float().to(device)
                    mask = mask_tiles.squeeze().to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculatio mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))



            if min_loss > (test_loss / len(val_loader)):
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss / len(val_loader))))
                min_loss = (test_loss / len(val_loader))
                # decrease += 1
                print('saving model...')
                torch.save(model.state_dict(), 'Unet_model-{:.3f}.pth'.format(val_iou_score / len(val_loader)))
                # if decrease % 5 == 0:
                #     print('saving model...')
                #     torch.save(model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score / len(val_loader)))

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

            for j in range(2):
                img, mask = next(iter(train_loader))
                img = img.detach().cpu()[0]
                mask = mask.detach().cpu()[0]
                # pred_mask.detach().cpu().numpy()
                img2 = np.transpose(img, (1, 2, 0))
                res = img[None, :, :, :]
                res11 = model(res.to(device).float())
                img_out = torch.softmax(res11.squeeze(), dim=0)
                mask_res = np.argmax(img_out.detach().cpu().numpy(), axis=0)
                display(img2, mask, mask_res, e)

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print("Epoch:{}/{}..".format(e + 1, epochs),
                  "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                  "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                  "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                  "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                  "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                  "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                  "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_miou': train_iou, 'val_miou': val_iou,
               'train_acc': train_acc, 'val_acc': val_acc,
               'lrs': lrs}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


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
        # print(f'class: {i}')
        # Используем заданный цвет для каждого класса
        color = colors[i]
        # Применяем маску и цвет для каждого класса
        color_mask[mask == i] = color


    return color_mask


def plot_loss(history):
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_score(history):
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('Score per epoch')
    plt.ylabel('mean IoU')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def plot_acc(history):
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()

def main():
    print(torch.__version__)
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
    train_path = 'train/'
    images_train, _, coco_train, classes_train = filterDataset(ann_file_name=ann_file_name,
                                                               percent_valid=0,
                                                               path_folder=train_path,
                                                               shuffie=False
                                                               )

    cat_ids = coco_train.getCatIds(classes_train)
    input_image_size = (256, 256)
    print(f"Number of training images: {len(images_train)}")

    path_dataset = os.path.join(DATASET_PATH, train_path)

    train_data = ImageData(annotations=coco_train,
                           image_list=images_train,
                           cat_ids=cat_ids,
                           root_path=path_dataset,
                           transform=True,
                           input_image_size=input_image_size
                           )
    BATCH_SIZE = 2
    train_dl = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    # for j in range(1):

    unet = UNet(n_channels=3, n_classes=4)
    unet.to(device)

    max_lr = 1e-3
    epoch = 15
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.3, 1.0, 1.0, 1.0]).to(device))
    optimizer = torch.optim.AdamW(unet.parameters(), lr=max_lr, weight_decay=weight_decay)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dl))

    history = fit(epoch, unet, train_dl, train_dl, criterion, optimizer, sched)

    torch.save(unet, 'Unet-Mobilenet.pth')

    plot_loss(history)
    plot_score(history)
    plot_acc(history)




def tensor2numpy(tensor):
    arr = np.array(tensor)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr


def display(img=None, mask=None, pred=None, epoch = None):
    ncols_num = 0
    if img is not None:
        ncols_num += 1
    if mask is not None:
        ncols_num += 1
    if pred is not None:
        ncols_num += 1
    fig, ax = plt.subplots(ncols=ncols_num, nrows=1, figsize=(5, 5))
    fig.suptitle(f'epoch: {epoch}', fontsize=20, fontweight='bold')
    ax[0].set_title('Фото')
    ax[1].set_title('маска')
    if pred is not None:
        ax[2].set_title('пред. маска')

    ax[0].imshow(tensor2numpy(img))
    ax[1].imshow(colorize_mask(tensor2numpy(mask)))
    if pred is not None:
        ax[2].imshow(colorize_mask(tensor2numpy(pred)))

    plt.show()


class ImageData(Dataset):
    def __init__(
            self,
            annotations: COCO,
            image_list: list[int],
            cat_ids: list[int],
            root_path: Path,
            transform,
            input_image_size=(16, 16)

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

        # IMAGE_SIZE = [256, 256]
        if random.random() > 0.5:
            params = tf.RandomResizedCrop.get_params(img1, scale=[0.5, 1.0], ratio=[0.75, 1.33])
            img1 = tf.functional.resized_crop(img1, *params, size=self.input_image_size, antialias=True)
            img2 = tf.functional.resized_crop(img2, *params, size=self.input_image_size, antialias=True)

        if random.random() > 0.5:
            img1 = tf.functional.hflip(img1)
            img2 = tf.functional.hflip(img2)

        autocontraster = T.RandomAutocontrast()
        img1 = autocontraster(img1)
        if random.random() > 0.5:
            img1 = tf.functional.vflip(img1)
            img2 = tf.functional.vflip(img2)

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
