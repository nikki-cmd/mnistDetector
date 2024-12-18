import os
import random

# Путь к папке с тестовыми изображениями
test_images_path = "C:/Users/kukus/OneDrive/Рабочий стол/AI/mnistDetector/test"

# Список всех файлов в папке
test_image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

for i in range(0, len(test_image_files)):
	os.rename(f"C:/Users/kukus/OneDrive/Рабочий стол/AI/mnistDetector/test/{test_image_files[i]}", 'C:/Users/kukus/OneDrive/Рабочий стол/AI/mnistDetector/test/' + str(random.randint(1, 99999999)) + '.jpg')