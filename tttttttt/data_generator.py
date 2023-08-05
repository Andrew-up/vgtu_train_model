import tensorflow as tf

class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self):
        super().__init__()

