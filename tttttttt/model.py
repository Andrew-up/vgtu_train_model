import tensorflow as tf

def get_model2(input_shape):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    # Создание CNN модели
    model = Sequential()

    # Добавление сверточного слоя
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', input_shape=input_shape))
    # Добавление слоя max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Добавление сверточного слоя
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(64, kernel_size=(1, 1), activation='relu'))
    # Добавление слоя max poolig
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Добавление сверточного слоя
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(128, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # # Добавление сверточного слоя
    # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    # # model.add(Conv2D(256, kernel_size=(1, 1), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # Преобразование изображения в вектор
    model.add(Flatten())

    # Добавление полносвязного слоя
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    # Выходной слой
    model.add(Dense(2, activation='linear'))
    return model

def get_model(input_shape):
    model = tf.keras.Sequential([
        # Сверточный слой
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.15),

        # Сверточный слой
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.15),

        # Сверточный слой
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.15),

        # Сглаживание
        tf.keras.layers.Flatten(),

        # Полносвязный слой
        tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),

        # Выходной слой
        tf.keras.layers.Dense(2)
    ])
    return model