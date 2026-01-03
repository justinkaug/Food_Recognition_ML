from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="food-recognition-ml",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning models for food classification, recognition, and nutrition label detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Food_Recognition_ML",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "wandb": ["wandb>=0.15.0"],
        "tensorflow": ["tensorflow>=2.13.0", "onnx-tf>=1.10.0"],
    },
    entry_points={
        "console_scripts": [
            "food-train-classifier=scripts.train_classifier:main",
            "food-train-recognizer=scripts.train_recognizer:main",
            "food-evaluate=scripts.evaluate:main",
            "food-export=scripts.export_models:main",
        ],
    },
)
