from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from matplotlib import pyplot as plt

# Путь к вашему датасету
dataset_path = "C:/Users/kukus/OneDrive/Рабочий стол/AI/mnistDetector/dataset"

# Трансформации данных
datagen = ImageDataGenerator(
    rescale=1.0/255.0,# Нормализация изображений
    validation_split=0.2 # 20% для валидации
)

# Генераторы для тренировочной и валидационной выборок
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),  # Размер изображений
    color_mode="rgb", # Серый цвет
    batch_size=32,          # Размер батча
    class_mode="categorical",
    subset="training"       # Только обучающая выборка
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=32,
    class_mode="categorical",
    subset="validation"     # Только валидационная выборка
)

classes = [0, 1, 2, 3]
num_filters = 8
filter_size = 10
pool_size = 2


model = Sequential([
    Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu', input_shape=(128, 128, 3)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size, strides=2, padding='valid'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Количество классов из генератора
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)


# Путь к папке с тестовыми изображениями
test_images_path = "C:/Users/kukus/OneDrive/Рабочий стол/AI/mnistDetector/test"

# Список всех файлов в папке
test_image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Предобработка изображений
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128), color_mode="rgb")  # Изменение размера и цвета
    img_array = img_to_array(img) / 255.0  # Нормализация
    return img_array

# Загрузка и обработка всех изображений
test_images = np.array([preprocess_image(os.path.join(test_images_path, file)) for file in test_image_files])

# Получение предсказаний
predictions = model.predict(test_images)
# Словарь классов
d = {0: 'car', 1: 'cat', 2: 'tree', 3: 'plane'}

# Вывод результатов
for i, prediction in enumerate(predictions):
    predicted_class = np.argmax(prediction)
    class_name = d.get(predicted_class, "Unknown")
    print(f"Изображение: {test_image_files[i]}, Предсказанный класс: {class_name}")
    plt.imshow(test_images[i])
    plt.title(f"Класс: {class_name}")
    plt.axis('off')
    plt.show()

