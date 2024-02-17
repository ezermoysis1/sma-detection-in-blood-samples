from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class my_ResNet_CNN(nn.Module):
    def __init__(self, dropout=0.5, aggregation='max', grayscale=0):
        super().__init__()

        # Initial convolutional layer to handle grayscale images
        self.grayscale = grayscale
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        # Use the first layers of ResNet-50 (exclude last 3 layers - children)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            *list(resnet.children())[:5],
        )  # Take children 0-6
        self.dropout = dropout
        self.aggregation = aggregation

        # Freeze the ResNet-50 weights
        for param in self.features.parameters():
            param.requires_grad = True

        self.fc2 = nn.Linear(16*16*1024, 1000)
        # Dropout layer with 50% probability
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(1000, 1)  # Adjusted for ResNet-50 output size
        self.sigmoid = nn.Sigmoid()

        # Attention weights and bias -- if 1000
        self.attention_weights = nn.Linear(1000, 1)
        self.attention_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode='train', bag_size=20, get_features=False):
        imgs_features = []

        if bag_size > 0:
            # Randomly choose bag_size=20 images from each bag of images in each training cycle
            if mode == 'train':
                idxs = random.sample(range(len(x)), bag_size)
                x = [x[i] for i in idxs]

        # Independently pass each image (or a random subset) of a bag through the layers
        for img in x:

            if self.grayscale == 1:
                # Adjust grayscale image to ResNet-50's input
                img = self.conv1(img)

            # Pass through ResNet-50 layers
            img_features = self.features(img)
            # Flatten
            img_features = img_features.view(img_features.size(0), -1)
            if mode == 'train':
                # Apply dropout on the middle layer
                img_features = self.dropout(img_features)
            # First linear layer from 8*8*1024 to 2048
            img_features = self.fc2(img_features)
            if mode == 'train':
                # Apply dropout on the last layer
                img_features = self.dropout(img_features)

            # Store in a list all the 2048 sized feature vectors for each sample
            imgs_features.append(img_features)

        # Apply max pooling across all images
        pooled_features = torch.stack(imgs_features)

        if self.aggregation == 'attention':
            # Attention pooling
            attention_weights = self.attention_weights(
                pooled_features,
            )  # Calculate attention weights
            # Apply softmax to get attention probabilities
            attention_weights = F.softmax(attention_weights, dim=0)
            pooled_features = pooled_features * attention_weights  # Apply attention weights
            # Sum the attention-weighted features
            pooled_features = torch.sum(pooled_features, dim=0)

        elif self.aggregation == 'mean':
            pooled_features = torch.mean(pooled_features, dim=0)

        elif self.aggregation == 'max':
            pooled_features, _ = torch.max(pooled_features, dim=0)

        # Flatten the pooled features
        pooled = pooled_features.view(pooled_features.size(0), -1)

        # Pass through the fully connected layers
        feature_vector = self.fc3(pooled)
        x = self.sigmoid(feature_vector)  # Apply sigmoid activation

        if get_features:
            return x, pooled  # Return both predictions and vector features
        else:
            return x  # Only return predictions
