# Advanced Image Detection and Segmentation with UNet and MNISTDD-RGB

This repository contains my coursework for CMPUT 328, focusing on advanced object detection and semantic segmentation techniques applied to the MNIST Double Digits RGB dataset. Utilizing the UNet architecture, this project tackles the challenges of detecting and segmenting overlapping digits in a synthetic dataset.

## Project Overview

The MNIST Double Digits RGB dataset poses unique challenges due to the presence of overlapping digits and varied background noise. This project demonstrates the application of UNet, a powerful convolutional neural network known for its effectiveness in semantic segmentation tasks. The goal is to accurately identify and segment each digit, distinguishing between closely positioned numerical digits.

## Features

- **Object Detection**: Implements bounding box detection to locate each digit within the image.
- **Semantic Segmentation**: Applies pixel-wise classification to accurately segment each digit.
- **Instance Segmentation**: Differentiates between overlapping digits using unique segmentation masks for each instance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install:

```bash
python >= 3.6
numpy
matplotlib
torch
torchvision
```

### Installing

A step by step series of examples that tell you how to get a development environment running:
1. Clone the repo:
   
      git clone https://github.com/anurag1102990/Object-Detection-and-Semantic-Segmentation.git

### Usage

To run the project:

      python main.py
      
This script will load the data, initialize the UNet model, perform training, and display the results. Adjust parameters within main.py to experiment with different configurations.

## Results

The UNet model trained on the MNIST Double Digits RGB dataset demonstrated robust performance across multiple metrics, underscoring its effectiveness in handling complex image segmentation and detection tasks:

- **Classification Accuracy**: Achieved a remarkable 93% accuracy in classifying the correct digits within the images.
- **Detection IOU (Intersection Over Union)**: Reached an impressive 82% in IOU, indicating a high degree of accuracy in detecting the correct bounding boxes around each digit.
- **Segmentation Accuracy**: Attained 75% accuracy in pixel-wise segmentation, effectively distinguishing individual digits and segmenting them from the background and each other, even in cases of overlap.

These results showcase the capability of the system to not only identify and locate digits within images but also to precisely segment them, which is critical in applications requiring fine-grained analysis of visual data.
