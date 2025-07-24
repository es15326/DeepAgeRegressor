# resnext_improved.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base import BaseNet
from loss import soft_aar_loss
from loss import *

class ResNext(BaseNet):
    """
    An age estimation model using the ResNeXt-50 32x4d architecture,
    loaded using the modern torchvision API.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # 1. Load the model using the current, recommended torchvision API
        if pretrained:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.resnext50_32x4d(weights=weights)

        # 2. Replace the final layer for the age regression task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

        # 3. Set the loss function
        # self.loss_func = nn.MSELoss() # Using a standard loss as a placeholder
        self.loss_func = weighted_mse_loss # Or your custom loss function
