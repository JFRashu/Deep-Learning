# Deep Learning Projects

This repository contains my deep learning projects and practice notebooks from my DL course. The goal is to document my learning journey and showcase implementations of various neural network models.

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
This repository serves as a collection of my deep learning experiments, primarily using PyTorch. Each project includes Jupyter notebooks with code, explanations, and results. The datasets used (e.g., FashionMNIST) are sourced from public repositories or torchvision.

## Projects

### FashionMNIST ANN Classifier
- **Description**: A neural network built with PyTorch to classify images from the FashionMNIST dataset (10 clothing categories). The model uses a fully connected ANN with ReLU activations and achieves ~81.7% test accuracy after 50 epochs.
- **Notebook**: [notebooks/fashionmnist-using-ann.ipynb](notebooks/fashionmnist-using-ann.ipynb)
- **Dataset**: FashionMNIST (60,000 training images, 10,000 test images, 28x28 grayscale).
- **Model Architecture**:
  - Input: 28x28 flattened images (784 features)
  - Layers: Linear(784, 250) → ReLU → Linear(250, 125) → ReLU → Linear(125, 75) → ReLU → Linear(75, 10)
  - Optimizer: SGD (learning rate = 0.001)
  - Loss: CrossEntropyLoss
- **Results**:
  - Test Accuracy: 81.7% (after 50 epochs)
  - Average Test Loss: 0.514
- **Visualizations**: Includes sample images from FashionMNIST with predicted labels.

## Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/DeepLearningProjects.git
   cd DeepLearningProjects
