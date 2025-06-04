# Deep Learning Projects

This repository contains my deep learning projects and practice notebooks from my DL course. The goal is to document my learning journey and showcase implementations of neural network models using PyTorch.

## Table of Contents
- [Project Overview](#project-overview)
- [Projects](#projects)
  - [FashionMNIST ANN Classifier](#fashionmnist-ann-classifier)
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
  - Test Accuracy: 81.7% (after 50 epochs)
  - Average Test Loss: 0.514
- **Visualizations**: Displays sample images from FashionMNIST with predicted labels.

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/DeepLearningProjects.git
   cd DeepLearningProjects
