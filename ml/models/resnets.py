from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class CustomResnet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.base_model = resnet18(weights=None)
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, input):
        return self.base_model(input)