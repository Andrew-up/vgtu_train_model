import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

# MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model.h5")
# MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model2.h5")
MODEL_H5_PATH = os.path.join(ROOT_DIR, "model/model3.h5")
DATASET_PATH = os.path.join(ROOT_DIR, "data_injured/train")
ANNOTATION_FILE_PATH = os.path.join(ROOT_DIR, "data_injured/annotations/labels_my-project-name_2022-11-15-02-32-33.json")
TENSORBOARD_LOGS = os.path.join(ROOT_DIR, "logs")

DATASET_LABELS = ['Асептическое', 'Бактериальное', 'Гнойное']

