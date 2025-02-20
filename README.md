# Currency Denomination Classification

This project aims to classify currency denominations using a deep learning model trained with TensorFlow and Keras.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running Predictions](#running-predictions)
- [Utilities](#utilities)
- [License](#license)

## Overview
The project includes a convolutional neural network (CNN) for recognizing different currency denominations. The model is trained using an image dataset and can classify banknotes into predefined categories.

## Dataset
The dataset is organized into two folders:
- `dataset/Train/` - Contains training images categorized into folders per class.
- `dataset/Test/` - Contains validation images categorized similarly.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/currency-classifier.git
   cd currency-classifier
   ```
2. Install dependencies:
   ```sh
   pip install tensorflow numpy opencv-python
   ```

## Training the Model
To train the model, run:
```sh
python try.py
```
This script:
- Loads and augments the dataset.
- Defines a CNN architecture.
- Trains the model and saves the best version (`best_model.keras`).

## Running Predictions
After training, use the `detect.py` script to classify a currency image:
```sh
python detect.py
```
It prompts for an image path and returns the predicted denomination with confidence.

## Utilities
The `utils.py` script includes functions for dataset organization and cleaning:
```sh
python utils.py
```

## License
This project is licensed under the MIT License.
