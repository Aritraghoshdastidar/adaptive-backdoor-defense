# core/models.py
import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18(num_classes=10):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
