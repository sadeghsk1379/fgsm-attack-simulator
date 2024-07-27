# CIFAR-10 Image Classification with ResNet18

This project demonstrates how to use a pre-trained ResNet18 model for image classification on the CIFAR-10 dataset. The model is fine-tuned to classify images into one of the 10 classes in CIFAR-10.

## Features

- **Data Preprocessing:** The CIFAR-10 dataset is resized and normalized to match the input requirements of ResNet18.
- **Model Training:** The pre-trained ResNet18 model is fine-tuned on the CIFAR-10 training dataset.
- **Performance Evaluation:** Training and test losses are tracked and plotted to evaluate model performance.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sadeghsk1379/fgsm-attack-simulator.git

2. **Install the required packages:**
   ```bash
   pip install torch torchvision matplotlib numpy
3. **Usage:**
   ```bash
   python main.py
4. **Output:**
   The script will download the CIFAR-10 dataset, preprocess the data, and train the ResNet18 model.
   Training and test losses will be printed for each epoch.
   A plot of the training and test losses will be displayed at the end.
   
   

   
