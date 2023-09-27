
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F

class my_ResNet_CNN_gc(nn.Module):
    def __init__(self, dropout=0.5, aggregation='max', grayscale=0):
        super().__init__()
        
        # Initial convolutional layer to handle grayscale images
        self.grayscale=grayscale
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        # Use the first layers of ResNet-50 (exclude last 3 layers - children)
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:7]) # Take children 0-6 
        self.dropout=dropout
        self.aggregation = aggregation

        # Freeze the ResNet-50 weights
        # for param in self.features.parameters():
        #     param.requires_grad = False
        
        self.fc2 = nn.Linear(8*8*1024, 2048)
        self.dropout = nn.Dropout(dropout) # Dropout layer with 50% probability
        self.fc3 = nn.Linear(2048, 1)  # Adjusted for ResNet-50 output size
        self.sigmoid = nn.Sigmoid()

        # Attention weights and bias -- if needed
        self.attention_weights = nn.Linear(2048, 1)
        self.attention_bias = nn.Parameter(torch.zeros(1))

        # GradCAM related
        self.gradients = None
        self.activations = None
        # Register hooks
        self.features[6].register_forward_hook(self.forward_hook_fn)
        self.features[6].register_backward_hook(self.backward_hook_fn)

    def forward(self, x, mode = 'train', bag_size=20):
        imgs_features = []

        if bag_size > 0:
            # Randomly choose bag_size=20 images from each bag of images in each training cycle
            if mode=='train':
                idxs=random.sample(range(len(x)),bag_size)
                x=[x[i] for i in idxs]

        # Independently pass each image (or a random subset) of a bag through the layers
        for img in x:

            if self.grayscale==1:
                # Adjust grayscale image to ResNet-50's input
                img = self.conv1(img)

            # Pass through ResNet-50 layers
            img_features = self.features(img)
            # Flatten
            img_features = img_features.view(img_features.size(0), -1)
            # Apply dropout on the middle layer
            img_features = self.dropout(img_features)
            # First linear layer from 8*8*1024 to 2048
            img_features=self.fc2(img_features)
            # Apply dropout on the last layer
            img_features = self.dropout(img_features)

            # Store in a list all the 2048 sized feature vectors for each sample
            imgs_features.append(img_features)

        # Apply max pooling across all images
        pooled_features = torch.stack(imgs_features)

        if self.aggregation == 'attention':
            # Attention pooling
            attention_weights = self.attention_weights(pooled_features)  # Calculate attention weights
            attention_weights = F.softmax(attention_weights, dim=0)  # Apply softmax to get attention probabilities
            pooled_features = pooled_features * attention_weights  # Apply attention weights
            pooled_features = torch.sum(pooled_features, dim=0)  # Sum the attention-weighted features

        elif self.aggregation == 'mean':
            pooled_features = torch.mean(pooled_features, dim=0)

        elif self.aggregation == 'max':
            pooled_features, _ = torch.max(pooled_features, dim=0)

        # Flatten the pooled features
        x = pooled_features.view(pooled_features.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc3(x)
        x = self.sigmoid(x)  # Apply sigmoid activation

        return x
    
    def forward_hook_fn(self, module, input, output):
        self.activations = output

    def backward_hook_fn(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute_gradcam_plus(self, input_tensor, class_idx=None):
        # Forward pass
        model_output = self.forward(input_tensor.unsqueeze(0), mode='test')
        if class_idx is None:
            class_idx = torch.argmax(model_output).item()

        # Backward pass
        model_output[0][class_idx].backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        weights = (alpha * gradients).mean(dim=(2, 3), keepdim=True)

        grad_cam_plus = F.relu((weights * activations).sum(dim=1)).squeeze(0)
        
        # Normalize the heatmap
        heatmap_min = torch.min(grad_cam_plus)
        heatmap_max = torch.max(grad_cam_plus)
        grad_cam_plus = (grad_cam_plus - heatmap_min) / (heatmap_max - heatmap_min)
        
        return grad_cam_plus