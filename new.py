import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO

output_dir = "rofl_test"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def download_images(query, max_images):
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")
    
    count = 0
    for img_tag in img_tags:
        if count >= max_images:
            break
        try:
            img_url = img_tag["src"]
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            process_image(img, f"{output_dir}/{query}_{count}.jpg")
            count += 1
        except Exception as e:
            print(f"исключение {e}")


def process_image(img, save_path):
    img = img.resize((128, 128))
    img.save(save_path)
    print(f"Сохранено: {save_path}")

# Используем функции
queries = ["топ лучшых мемов 2013"]  # Поисковые запросы
for query in queries:
    download_images(query, max_images=10000)
