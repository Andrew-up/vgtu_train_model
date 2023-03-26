import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

# MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model.h5")
# MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model2.h5")
# MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model3.h5")
MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/")
MODEL_H5_FILE_NAME = os.path.join(ROOT_DIR, "model3.h5")
# DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/image")
# DATASET_PATH = os.path.join(ROOT_DIR, "data_injured/train")
# ANNOTATION_FILE_PATH = os.path.join(ROOT_DIR, "data_injured/annotations/labels_my-project-name_2022-11-15-02-32-33.json")
ANNOTATION_FILE_PATH = os.path.join(ROOT_DIR, "dataset/annotations/data.json")
# ANNOTATION_FILE_PATH_TRAIN = os.path.join(ROOT_DIR, "data/ann/data.json")
ANNOTATION_FILE_PATH_TRAIN = os.path.join(ROOT_DIR, "dataset/train/_annotations.coco.json")
ANNOTATION_FILE_PATH_VALID = os.path.join(ROOT_DIR, "dataset/valid/_annotations.coco.json")
ANNOTATION_FILE_PATH_TEST = os.path.join(ROOT_DIR, "dataset/test/_annotations.coco.json")
TENSORBOARD_LOGS = os.path.join(ROOT_DIR, "logs")


