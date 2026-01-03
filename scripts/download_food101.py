"""Download Food-101 dataset"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.download import download_food101


def main():
    """Download Food-101 dataset to data/raw directory"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Starting Food-101 dataset download...")
    print(f"Destination: {data_dir}")
    
    try:
        download_food101(data_dir)
        print("\nDownload completed successfully!")
        print(f"Dataset location: {os.path.join(data_dir, 'food-101')}")
    except Exception as e:
        print(f"\nError during download: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
