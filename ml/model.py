import torch
import torch.nn as nn
from torchvision import models

class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeResNet, self).__init__()
        # Load pre-trained ResNet50
        # If pretrained=True, it downloads weights from ImageNet
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        
        # We need to change the final fully connected layer
        # ResNet50 has 2048 in_features for the fc layer
        num_features = self.model.fc.in_features
        
        # Replace it with a new layer for our 2 classes (REAL and FAKE)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.model(x)

    def get_target_layer(self):
        # We use the last convolutional layer output for Grad-CAM
        return [self.model.layer4[-1]]
