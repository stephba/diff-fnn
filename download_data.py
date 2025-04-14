import os
import requests
from zipfile import ZipFile

MOVIELENS_PATH = "data/movielens1m/raw"
ANIME_PATH = "data/anime/raw"

def download_and_extract_zip(url, output_dir):
    temp_file = 'temp.zip'

    print("Download zip file...")
    response = requests.get(url)
    with open(temp_file, 'wb') as file:
        file.write(response.content)
    print("done.")

    print("Extract zip file...")
    os.makedirs(output_dir, exist_ok=True)
    with ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("done.")

    print("Delete zip file...")
    os.remove(temp_file)
    print("done.")

def download_movielens_1m():
    print("Download MovieLens 1M dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    download_and_extract_zip(url, MOVIELENS_PATH)

    print("Remove nested folder...")
    old_folder_path = os.path.join(MOVIELENS_PATH, 'ml-1m')
    for filename in os.listdir(old_folder_path):
        src_path = os.path.join(old_folder_path, filename)
        dst_path = os.path.join(MOVIELENS_PATH, filename)
        os.rename(src_path, dst_path)
    os.rmdir(old_folder_path) # delete empty folder
    print("done.")

    print("Download MovieLens 1M dataset done.")

def download_anime():
    print("Download Anime dataset...")
    url = 'https://zenodo.org/records/7428435/files/anime.zip'

    download_and_extract_zip(url, ANIME_PATH)

    print("Download Anime dataset done.")

if __name__ == "__main__":
    download_movielens_1m()
    download_anime()
