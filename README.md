# Deep Learning Projects

This repository contains my deep learning projects and practice notebooks from my DL course. The goal is to document my learning journey and showcase implementations of neural network models using PyTorch.

## Table of Contents
- [Project Overview](#project-overview)
- [Projects](#projects)
  - [FashionMNIST ANN Classifier](#fashionmnist-ann-classifier)
  - [Website Classification ANN](#website-classification-using-ann)
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

## Project Overview
This repository is a collection of my deep learning experiments, primarily using PyTorch. Each project includes Jupyter notebooks with code, explanations, and results. The datasets, such as FashionMNIST, are sourced from `torchvision`.

## Projects

### FashionMNIST ANN Classifier
- **Description**: A neural network built with PyTorch to classify images from the FashionMNIST dataset (10 clothing categories). The model is a fully connected ANN with ReLU activations, trained for 50 epochs, achieving ~81.7% test accuracy.
- **Notebook**: [fashionmnist-using-ann.ipynb](fashionmnist-using-ann.ipynb)
- **Dataset**: FashionMNIST (60,000 training images, 10,000 test images, 28x28 grayscale).
- **Model Architecture**:
  - Input: 28x28 flattened images (784 features)
  - Layers: Linear(784, 250) → ReLU → Linear(250, 125) → ReLU → Linear(125, 75) → ReLU → Linear(75, 10)
  - Optimizer: SGD (learning rate = 0.001)
  - Loss: CrossEntropyLoss
- **Results**:
  - Test Accuracy: 87.0% (after 5 epochs)
  - Average Test Loss: 0.514
- **Visualizations**: Displays sample images from FashionMNIST with predicted labels.

### Website Classification ANN

- **Description**: A PyTorch-based ANN to classify websites into 16 categories (e.g., Sports, Education, Business) using text features from CountVectorizer. Uses a deeper 5-layer ANN with ReLU activations and dropout to prevent overfitting, trained for 50 epochs.
- **Notebook**: [website-classification-using-ann.ipynb](website-classification-using-ann.ipynb)
- **Dataset**: Website Classification from Kaggle (\~1400 samples, 16 categories).
- **Model Architecture**:
  - Input: 5000 features (CountVectorizer)
  - Layers: Linear(5000, 512) → ReLU → Dropout(0.3) → Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 16)
  - Optimizer: SGD (learning rate = 0.01)
  - Loss: CrossEntropyLoss
- **Results**:
  - Test Accuracy:: 86.17%
  - Visualizations: Confusion matrix heatmap, classification report.
- **Test Cases**: Predicts categories for texts (e.g., cricket news, machine learning article, food delivery ad).

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/JFRashu/Deep-Learning.git
   cd DeepLearningProjects
