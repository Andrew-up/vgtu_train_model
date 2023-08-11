import os
import random
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.io as io
import torchvision.transforms as T
import torchvision.transforms as tf
import torchvision.transforms.functional as functional
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from controller_vgtu_train.model_history import ModelHistory
from controller_vgtu_train.subprocess_train_model_controller import get_last_model_history, update_model_history
from definitions import DATASET_PATH
from definitions import MODEL_PATH, DEFAULT_MODEL_NAME, ROOT_DIR
from utilits.unet_pytorch import UNet


def rename_output_file(old_path, new_path):
    if os.path.exists(old_path):
        os.replace(old_path, new_path)
    return new_path


def torch_to_onnx_to_tflite(batch_size=4,
                            path_pth_file=None,
                            path_onnx_file=None,
                            path_tflite_file=None,
                            result_onnx=None,
                            image_size=(128, 128),
                            n_classes=3):
    import onnx2tf
    torch.cuda.empty_cache()
    model = UNet(n_classes=n_classes + 1, n_channels=3)
    x = torch.randn(1, 3, image_size[0], image_size[1])
    model.load_state_dict(torch.load(path_pth_file, map_location='cpu'))
    model.eval()

    torch.onnx.export(model,
                      x,
                      path_onnx_file,
                      export_params=True,
                      verbose=True,
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}}
                      )

    onnx2tf.convert(
        input_onnx_file_path=path_onnx_file,
        output_folder_path=MODEL_PATH,
        copy_onnx_input_output_names_to_tflite=True,
        output_nms_with_dynamic_tensor=True,
    )

    rename_output_file(os.path.join(MODEL_PATH, result_onnx), path_tflite_file)


def filterDataset(ann_file_name, classes=None, mode='train', percent_valid=50, path_folder=None, shuffie=True):
    weight_list = [0.3]
    # initialize COCO api for instance annotations
    annFile = os.path.join(DATASET_PATH, ann_file_name)
    annFile = os.path.normpath(annFile)
    if not os.path.exists(annFile):
        return [], None, None, None, None

    coco = COCO(annFile)
    images = []
    if classes != None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    if classes is None:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)

    categories = coco.loadCats(coco.getCatIds())
    # перебираем категории
    for category in categories:
        # получаем id категории
        category_id = category['id']
        # получаем аннотации для данной категории
        ann_ids = coco.getAnnIds(catIds=[category_id])
        # если есть аннотации, то добавляем 1 в список, иначе 0
        if len(ann_ids) > 0:
            weight_list.append(1.0)
        else:
            weight_list.append(0.0)

    group_class = []

    for i in classes:
        l = []
        catIds = coco.getCatIds(catNms=i)
        imgssss = coco.getImgIds(catIds=catIds)
        l += coco.loadImgs(imgssss)
        valid_files = []
        for image_one in l:
            imagePath = os.path.join(DATASET_PATH, path_folder, image_one['file_name'])
            print(imagePath)
            imagePath = os.path.normpath(imagePath)
            if os.path.exists(imagePath):
                valid_files.append(image_one)
            else:
                print(f'no image : {imagePath}')
        group_class.append(valid_files)

    images_train_tmp = []
    images_valid_tmp = []

    images_train_unique = []
    images_valid_unique = []
    for classesss in group_class:
        if percent_valid > 0:
            b = round(percent_valid / 100 * len(classesss))
            images_train_tmp += classesss[b:]
            images_valid_tmp += classesss[:b]
        else:
            images_train_tmp += classesss

    for i in range(len(images_train_tmp)):
        if images_train_tmp[i] not in images_train_unique:
            images_train_unique.append(images_train_tmp[i])

    for i in range(len(images_valid_tmp)):
        if images_valid_tmp[i] not in images_valid_unique:
            images_valid_unique.append(images_valid_tmp[i])

    if shuffie:
        random.shuffle(images_train_unique)
        random.shuffle(images_valid_unique)

    if classes is not None:
        return images_train_unique, images_valid_unique, coco, classes, weight_list
    else:
        classes = list()
        for i in coco.cats:
            name = coco.cats[i]['name']
            classes.append(name)
        return images_train_unique, images_valid_unique, coco, classes, weight_list


n_fold = 5
pad_left = 27
pad_right = 27
fine_size = 202
epoch = 30
snapshot = 6
max_lr = 0.012
min_lr = 0.001
momentum = 0.9
weight_decay = 1e-4

device = torch.device('cuda')

file_info = {
    'name_file': None,
    'date': None,
    'path': None
}


def delete_legacy_models_and_zip(max_files_legacy: int):
    list_pth: file_info = []
    list_tflite: file_info = []
    list_zip: file_info = []
    sum_file_delete = 0
    for root, dirs, files in os.walk(MODEL_PATH):
        for file in files:
            if file.endswith('.pth'):
                a = os.stat(os.path.join(MODEL_PATH, file))
                created = time.ctime(a.st_atime)
                list_pth.append({'name_file': file, 'date': datetime.strptime(created, '%c'),
                                 'path': os.path.join(MODEL_PATH, file)})
            if file.endswith('.tflite'):
                a = os.stat(os.path.join(MODEL_PATH, file))
                created = time.ctime(a.st_atime)
                # print(type(created))
                list_tflite.append({'name_file': file, 'date': datetime.strptime(created, '%c'),
                                    'path': os.path.join(MODEL_PATH, file)})

            if file.endswith('.zip'):
                a = os.stat(os.path.join(MODEL_PATH, file))
                created = time.ctime(a.st_atime)
                # print(type(created))
                list_zip.append({'name_file': file, 'date': datetime.strptime(created, '%c'),
                                 'path': os.path.join(MODEL_PATH, file)})

    newlist_pth = sorted(list_pth, key=lambda d: d['date'], reverse=False)
    newlist_zip = sorted(list_zip, key=lambda d: d['date'], reverse=False)
    newlist_tflite = sorted(list_tflite, key=lambda d: d['date'], reverse=False)

    if len(newlist_pth) > max_files_legacy:
        summ_deletefiles = len(newlist_pth) - max_files_legacy
        for i in newlist_pth[0:summ_deletefiles]:
            if os.path.exists(i['path']):
                os.remove(i['path'])
                sum_file_delete += 1
                print(f"REMOVE FILE: {i['path']}")

    if len(newlist_tflite) > max_files_legacy:
        summ_deletefiles = len(newlist_tflite) - max_files_legacy
        for i in newlist_tflite[0:summ_deletefiles]:
            if os.path.exists(i['path']):
                os.remove(i['path'])
                sum_file_delete += 1
                print(f"REMOVE FILE: {i['path']}")

    if len(newlist_zip) > max_files_legacy:
        summ_deletefiles = len(newlist_zip) - max_files_legacy
        for i in newlist_zip[0:summ_deletefiles]:
            print(i['path'])
            if os.path.exists(i['path']):
                os.remove(i['path'])
                sum_file_delete += 1
                print(f"REMOVE FILE: {i['path']}")
    return sum_file_delete


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
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def fit(epochs,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        save_path_model=None,
        model_history: ModelHistory = None,
        len_classes=1):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    not_improve = 0

    if model_history:
        model_history.quality_dataset = len(train_loader) + len(val_loader)
        model_history.quality_train_dataset = len(train_loader)
        model_history.quality_valid_dataset = len(val_loader)
        model_history.total_epochs = epochs
        model_history.status = 'train'
        update_model_history(model_history)

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
            image = image_tiles.float().to(device)
            mask = mask_tiles.squeeze().to(device)

            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask, len_classes)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

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
                    image_tiles, mask_tiles = data
                    image = image_tiles.float().to(device)
                    mask = mask_tiles.squeeze().to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask, len_classes)
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
                print('saving model...')
                torch.save(model.state_dict(), save_path_model)

                model_history.current_epochs = e + 1
                update_model_history(model_history)

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = (test_loss / len(val_loader))
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == 7:
                    print('Loss not decrease for 7 times, Stop Training')
                    break

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


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    timer = time.time()
    check_garbage_files_count = delete_legacy_models_and_zip(max_files_legacy=10)
    if check_garbage_files_count == 0:
        print('Мусора нет')
    else:
        print(f'Удалено старых моделей h5 и zip архивов: {check_garbage_files_count}')

    name = 'default'
    model_history = get_last_model_history()
    if model_history.path_dataset:
        name = model_history.path_dataset

    print(model_history.__dict__)
    # return 0
    ann_file_name = Path(name + '/annotations/' + '/data.json')
    train_path = os.path.normpath(Path(name + '/image/'))
    print(ann_file_name)
    print(train_path)
    BATCH_SIZE = 2
    images_train, _, coco_train, classes_train, weight_list = filterDataset(ann_file_name=ann_file_name,
                                                                            percent_valid=0,
                                                                            path_folder=train_path,
                                                                            shuffie=False
                                                                            )
    if len(images_train) < BATCH_SIZE:
        model_history.status = 'Ошибка: Датасет слишком маленький'
        update_model_history(model_history)
        return 0

    if not (images_train or coco_train or classes_train or weight_list):
        model_history.status = 'Ошибка: Фильтрации датасета'
        update_model_history(model_history)
        return 0

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

    train_dl = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    img, mask = train_data[0]
    input_shape = f'{list(reversed(img.shape))}'
    output_shape = f'[{img.shape[1]} {img.shape[2]} {len(classes_train) + 1}]'

    # return 0
    unet = UNet(n_channels=3, n_classes=len(classes_train) + 1)
    unet.to(device)

    max_lr = 1e-3
    # epoch = 15
    weight_decay = 1e-4

    # return 0
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weight_list).to(device))
    optimizer = torch.optim.AdamW(unet.parameters(), lr=max_lr, weight_decay=weight_decay)
    # steps_per_epoch = len(train_dl) * 3
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                                steps_per_epoch=len(train_dl))
    path_model = os.path.join(MODEL_PATH, DEFAULT_MODEL_NAME)
    model_onnx_file_name = 'model_onnx.onnx'
    path_model_onnx = os.path.join(MODEL_PATH, model_onnx_file_name)
    path_model_tflite = os.path.join(MODEL_PATH, 'model_tflite.tflite')

    model_tf_lite_name_new = ''

    if model_history:
        model_tf_lite_name_new = f'model_{model_history.version.replace(".", "_")}.tflite'
        path_model_tflite = os.path.join(MODEL_PATH, model_tf_lite_name_new)
        path_model = os.path.join(MODEL_PATH, model_history.name_file)

    history = fit(epoch,
                  unet,
                  train_dl,
                  train_dl,
                  criterion,
                  optimizer,
                  sched,
                  save_path_model=path_model,
                  model_history=model_history,
                  len_classes=len(classes_train))

    torch_to_onnx_to_tflite(path_pth_file=path_model,
                            path_onnx_file=path_model_onnx,
                            path_tflite_file=path_model_tflite,
                            image_size=input_image_size,
                            result_onnx='model_onnx_float16.tflite',
                            n_classes=len(classes_train))

    if model_history:
        model_history.date_train = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
        model_history.num_classes = str(len(classes_train))
        model_history.input_size = input_shape
        model_history.output_size = output_shape
        model_history.name_file = model_tf_lite_name_new
        model_history.status = 'completed'
        model_history.time_train = time.strftime("время обучения: %H часов %M минут %S секунд",
                                                 time.gmtime(time.time() - timer))
        update_model_history(model_history)

    path_zip = zipfile.ZipFile(f'{os.path.splitext(path_model)[0]}.zip', 'w')
    path_zip.write(path_model_tflite, arcname=f'{model_history.name_file}')
    path_zip.close()


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
        self.files = [str(root_path + '/' + img["file_name"]) for img in self.img_data]
        print(self.files)
        self.transform = transform
        self.input_image_size = input_image_size

    def __len__(self) -> int:
        return len(self.files)

    def edit_background(self, img, mask_image):
        img = np.transpose(img.detach().numpy(), (1, 2, 0))
        mask_image = np.transpose(mask_image.detach().numpy(), (1, 2, 0))

        dir = os.path.join(ROOT_DIR, "rand_back")
        l = os.listdir(dir)
        allfiles = []
        for file in l:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                allfiles.append(os.path.join(dir, file))
        path_img_back = random.choice(allfiles)
        w = img.shape[0]
        h = img.shape[1]

        background = cv2.imread(os.path.join(ROOT_DIR, path_img_back), flags=cv2.COLOR_BGR2RGB)
        background = cv2.resize(background, dsize=(w, h), interpolation=cv2.INTER_AREA)
        img_n = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)

        background_mask = np.logical_not(np.squeeze(mask_image))
        new_img = np.where(background_mask[..., None], background, img_n)
        new_img = np.transpose(new_img, (2, 1, 0))
        return torch.from_numpy(new_img)

    def train_transform(self,
                        image: torch.Tensor,
                        binary_mask: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:

        if random.random() > 0.5:
            params = tf.RandomResizedCrop.get_params(image, scale=[0.5, 1.0], ratio=[0.75, 1.33])
            image = tf.functional.resized_crop(image, *params, size=self.input_image_size, antialias=True)
            binary_mask = tf.functional.resized_crop(binary_mask, *params, size=self.input_image_size, antialias=True)

        if random.random() > 0.5:
            image = tf.functional.hflip(image)
            binary_mask = tf.functional.hflip(binary_mask)

        autocontrast = T.RandomAutocontrast()
        image = autocontrast(image)
        if random.random() > 0.5:
            image = tf.functional.vflip(image)
            binary_mask = tf.functional.vflip(binary_mask)

        if random.random() > 0.5:
            pass
            # image = self.edit_background(image, binary_mask)

        return image, binary_mask

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        ann_ids = self.annotations.getAnnIds(
            imgIds=self.img_data[i]['id'],
            catIds=self.cat_ids,
            iscrowd=None
        )
        annotations = self.annotations.loadAnns(ann_ids)
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"]
                                                 for ann in annotations]), axis=0)).unsqueeze(0)

        img = io.read_image(self.files[i])

        img = torchvision.transforms.Resize(self.input_image_size, antialias=True)(img)
        mask = torchvision.transforms.Resize(self.input_image_size, antialias=True)(mask)

        if img.shape[0] == 1:
            img = torch.cat([img] * 3)

        if self.transform:
            return self.train_transform(image=img, binary_mask=mask)

        return img, mask


if __name__ == "__main__":
    main()
