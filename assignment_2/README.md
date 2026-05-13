# Assignment 2: Two-Layer Neural Network

## Overview
This assignment extends the concepts from Assignment 1 to build a two-layer neural network (Multi-Layer Perceptron) with a ReLU hidden layer. 

Key implementations include:
- Forward and backward passes for a 2-layer network.
- Gradient verification using PyTorch.
- Training the network using Mini-Batch Gradient Descent.
- **Cyclical Learning Rates:** Implementing learning rate schedules that oscillate between `eta_min` and `eta_max` to help the network converge faster and better, avoiding local minima.
- **Hyperparameter Search:** Executing both coarse and fine random searches to find the optimal L2 regularization parameter $\lambda$.

## Results
The cyclical learning rate implementation showed significant improvements in training speed and validation accuracy. 
Through coarse and fine hyperparameter search (results recorded in `coarse_search_results.txt` and `fine_search_results.txt`), the optimal $\lambda$ was found, achieving a final validation accuracy of around 52% on the CIFAR-10 dataset.
