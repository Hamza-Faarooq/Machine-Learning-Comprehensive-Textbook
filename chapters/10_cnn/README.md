# Chapter 10: Convolutional Neural Networks

## Chapter Overview
This chapter covers spatial deep learning architectures. We explore why fully connected networks fail on high-dimensional visual inputs and introduce 2D cross-correlation, parameter sharing, and local receptive fields. We analyze landmark architectures from LeNet-5 to ResNets and Vision Transformers (ViT).

## Learning Objectives
* Calculate output feature dimensions, parameter counts, and receptive fields for arbitrary convolutional layers.
* Understand translation equivariance and parameter sharing.
* Implement residual connections to resolve the gradient degradation problem in very deep networks.
* Apply transfer learning strategies (feature extraction vs. full fine-tuning) on pretrained vision models.

## Concepts Covered
* 2D Cross-Correlation and Convolutions
* Stride, Padding, and Dilation
* Max Pooling, Average Pooling, and Global Average Pooling (GAP)
* Landmark Architectures: LeNet-5, AlexNet, VGG, Inception, ResNet, DenseNet
* Transfer Learning Workflows

## Connection to Textbook
The implementation maps directly to Chapter 10 of the textbook, demonstrating the exact residual block formulation $a^{[l+2]} = \text{ReLU}(\mathcal{F}(a^{[l]}) + a^{[l]})$ used to build deep ResNets without suffering from vanishing gradients.
