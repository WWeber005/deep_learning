# Assignment 4: Recurrent Neural Networks (RNN)

## Overview
In this assignment, the objective is to implement a vanilla Recurrent Neural Network (RNN) from scratch to synthesize text character by character. The model is trained on a continuous text file (`goblet_book.txt`, which contains the text of "Harry Potter and the Goblet of Fire").

The work involved:
- Parsing and preprocessing text data into a vocabulary and representing characters as one-hot vectors.
- Implementing the forward pass, computing the hidden states with `tanh` activations, and calculating the cross-entropy loss over sequences of characters.
- Deriving and computing **Backpropagation Through Time (BPTT)** analytically to calculate the exact gradients.
- Validating the custom gradient computations against PyTorch's auto-differentiation mechanism.
- Training the RNN using the **Adam optimization** algorithm.
- Applying **Gradient Clipping** to mitigate the exploding gradient problem commonly found in RNNs.
- Periodically synthesizing and sampling new text sequences during training to visually evaluate the model's progress in learning the language structure.

## Results
Throughout the training process, the model optimizes its weights, causing the "smooth loss" to decrease steadily. As training progresses, the generated text evolves from completely random characters into somewhat recognizable English words and syntactic structures mimicking the original book's style. 

The gradient tests verify that the analytical BPTT implementation is fully correct, showing a near-zero relative error when compared with PyTorch's `autograd`.
