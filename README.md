# Skin Lesion Detection CNN

## Overview

This project involves the development of a Convolutional Neural Network (CNN) designed to analyze images of skin lesions and assess the likelihood that they are melanomas, a type of skin cancer. The CNN has been trained to classify skin lesions as either benign or potentially malignant based on the provided dataset.

## Repository Contents

- **main.py**: This is the main application file. You can use it to input an image of a skin lesion, and the CNN will provide a prediction regarding its malignancy.

- **cnn_train.py**: This script was used to train the neural network on a labeled dataset of skin lesion images. It contains the training process and model architecture.

- **engine.py**: This script loads the pre-trained neural network model for making predictions. You can use it to perform predictions on new, unseen skin lesion images.

- **cnn.pth**: This file contains the pre-trained CNN model that you can use for making predictions without retraining.

- **quest/**: This directory is meant to hold the skin lesion images you want to analyze using the neural network. Place the images you want to evaluate here.

## Disclaimer

Please note that this program is not licensed, and its assessments should not be considered definitive or medically accurate. It is intended for educational and experimental purposes only. Any predictions made by the neural network should not be relied upon for medical decisions. Always consult with a qualified healthcare professional for skin health concerns and diagnoses.


