import tensorflow as tf
from tensorflow import keras
import glob
import numpy as np
import matplotlib.pyplot as plt

strategy = tf.distribute.MirroredStrategy()
batch_size_per_replica = 256
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
paths = glob.glob('./sample/*.jpg')

MAX_CAPTCHA = 4
CHAR_SET_LEN = len(alphabet)


def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = alphabet.index(c)
        vector[i][idx] = 1.0
    return vector


def preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [64, 168])
    image = tf.cast(image, tf.float32)
    image = image / 255.0 * 2 - 1
    return image, label


labels = [text2vec(item.split('/')[-1].split('.')[0]) for item in paths]
dataset_len = len(paths)
train_len = int(0.9 * dataset_len)
test_len = dataset_len - train_len

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.map(preprocess)
train_data = dataset.take(train_len).repeat().shuffle(train_len).batch(batch_size)
test_data = dataset.take(test_len).batch(batch_size)

epochs = 100
learning_rate = 0.001


# 定义学习率衰减函数
def scheduler(ep):
    if ep < epochs * 0.4:
        return learning_rate
    if ep < epochs * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


change_Lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

with strategy.scope():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3, 3), input_shape=(64, 168, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=2),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=2),

        keras.layers.Conv2D(256, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=2),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN),
        keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]),
        tf.keras.layers.Softmax()
    ])

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(lr=0.0001),
        metrics=['acc']
    )
model.summary()
model.fit(train_data, epochs=epochs, steps_per_epoch=train_len // batch_size, validation_data=test_data, validation_steps=1, callbacks=[change_Lr])
model.save('./captcha.h5')
img, lbl = next(iter(test_data))
print(lbl[0])
pred = model.predict(img)
print(tf.argmax(pred, 2))
plt.imshow((img[0] + 1)/2)
plt.show()
