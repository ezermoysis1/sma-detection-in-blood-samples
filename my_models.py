
import random
import torch
import torch.nn as nn
import torchvision.models as models



class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.5, attention=0):
        super().__init__()
        
        # Use the first layers of ResNet-50 --> Look a bit more into this
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:6]) # Check this(?)
        self.dropout=dropout
        self.attention = attention

        # Freeze the ResNet-50 weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.fc2 = nn.Linear(32768, 2048)
        self.dropout = nn.Dropout(dropout) # Dropout layer with 50% probability
        self.fc3 = nn.Linear(2048, 1)  # Adjusted for ResNet-50 output size
        self.sigmoid = nn.Sigmoid()

        # Attention weights and bias -- if needed
        self.attention_weights = nn.Linear(2048, 1)
        self.attention_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode = 'train', bag_size=20):
        imgs_features = []

        # Randomly choose bag_size=20 images from each bag of images in each training cycle
        if mode=='train':
            idxs=random.sample(range(len(x)),bag_size)
            x=[x[i] for i in idxs]

        # Independently pass each image (or a random subset) of a bag through the layers
        for img in x:
            # Pass through ResNet-50 layers
            img_features = self.features(img)
            # Flatten
            img_features = img_features.view(img_features.size(0), -1)
            # Apply dropout on the middle layer
            img_features = self.dropout(img_features)
            # First linear layer from 32768 to 2048
            img_features=self.fc2(img_features)
            # Apply dropout on the last layer
            img_features = self.dropout(img_features)

            # Store in a list all the 2048 sized feature vectors for each sample
            imgs_features.append(img_features)

        # Apply max pooling across all images
        pooled_features = torch.stack(imgs_features)

        if self.attention == 1:
            # Attention pooling
            attention_weights = self.attention_weights(pooled_features)  # Calculate attention weights
            attention_weights = F.softmax(attention_weights, dim=0)  # Apply softmax to get attention probabilities
            pooled_features = pooled_features * attention_weights  # Apply attention weights
            pooled_features = torch.sum(pooled_features, dim=0)  # Sum the attention-weighted features

        elif self.attention == 0:
            pooled_features, _ = torch.max(pooled_features, dim=0)

        # Flatten the pooled features
        x = pooled_features.view(pooled_features.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply sigmoid activation

        return x