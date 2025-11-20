import torch
import torch.nn as nn
from torchvision import models


class StyleClassifier(nn.Module):
    """
    Simple ResNet50-based classifier for WikiArt styles
    Number of output classes should match the number of style folders
    """
    def __init__(self, num_styles):
        super().__init__()

        # Load pretrained ResNet50 from torchvision
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Replace the final FC layer to match the number of classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_styles)

    def forward(self, x):
        return self.model(x)
