import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root


MODEL_PATH = os.path.join(ROOT_DIR, "model/")
DEFAULT_MODEL_NAME = os.path.join(ROOT_DIR, "model_unet.pth")
# DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
DATASET_PATH = os.path.join(ROOT_DIR, "dataset/")
# DATASET_PATH = os.path.join(ROOT_DIR, "data_injured/train")
# ANNOTATION_FILE_PATH = os.path.join(ROOT_DIR, "data_injured/annotations/labels_my-project-name_2022-11-15-02-32-33.json")
ANNOTATION_FILE_PATH = os.path.join(ROOT_DIR, "dataset/annotations/data.json")
ANNOTATION_FILE_PATH_TRAIN = os.path.join(ROOT_DIR, "data/ann/data.json")
# ANNOTATION_FILE_PATH_TRAIN = os.path.join(ROOT_DIR, "dataset/train/_annotations.coco.json")
# ANNOTATION_FILE_PATH_VALID = os.path.join(ROOT_DIR, "dataset/valid/data.json")
# ANNOTATION_FILE_PATH_VALID_IMAGE = os.path.join(ROOT_DIR, "dataset/valid/")
TENSORBOARD_LOGS = os.path.join(ROOT_DIR, "logs")


