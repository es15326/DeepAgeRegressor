# resnet_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base import BaseNet
from loss import weighted_mse_loss

class ResNetAgeModel(BaseNet):
    """
    An age estimation model using the ResNet-50 architecture.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # 1. Load the ResNet-50 model using the current torchvision API
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.resnet50(weights=weights)

        # 2. Replace the final layer for the age regression task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

        # 3. Set the loss function
        self.loss_func = nn.MSELoss() # Using a standard loss as a placeholder
        # self.loss_func = weighted_mse_loss # Or your custom loss function
