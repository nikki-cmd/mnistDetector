import os
from matplotlib import pyplot as plt
import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
TF_CPP_MIN_LOG_LEVEL = 2
TF_ENABLE_ONEDNN_OPTS = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis = 3)
test_images = np.expand_dims(test_images, axis = 3)
print(train_images.shape)
print(test_images.shape)


classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_filters = 8
filter_size = 3
pool_size = 2



model = Sequential([
    Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size, strides=2, padding='valid'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, to_categorical(train_labels), epochs=3, validation_data=(test_images, to_categorical(test_labels)))

predictions = model.predict(test_images[:10])

d={0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle Boot'}

for n in range(0, 10):
    print("Хотели", d[np.argmax(predictions[n])], "получили:", d[test_labels[n]])
    plt.imshow(test_images[n].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()

