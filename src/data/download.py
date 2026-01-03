"""Dataset download utilities"""

import os
import requests
import tarfile
from tqdm import tqdm


def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def download_food101(data_dir):
    """Download and extract Food-101 dataset"""
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    tar_path = os.path.join(data_dir, "food-101.tar.gz")
    
    print(f"Downloading Food-101 dataset...")
    download_file(url, tar_path)
    
    print(f"Extracting Food-101 dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    os.remove(tar_path)
    print(f"Food-101 dataset downloaded and extracted to {data_dir}")
