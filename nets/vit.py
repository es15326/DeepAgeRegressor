# vit_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base import BaseNet
# Assuming your custom loss function is in a file named loss.py
# from loss import weighted_mse_loss

class ViTAgeModel(BaseNet):
    """
    An age estimation model using the Vision Transformer (ViT-B/16) architecture.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Load a pretrained Vision Transformer model
        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.vit_b_16(weights=weights)

        # 2. Replace the final classification head for our regression task
        # Get the number of input features for the classifier head
        in_features = self.model.heads.head.in_features
        # The new head predicts a single value (age)
        self.model.heads.head = nn.Linear(in_features, 1)
        
        # 3. Set the loss function (can be your custom one)
        # self.loss_func = weighted_mse_loss
        self.loss_func = nn.MSELoss() # Using a standard loss as a placeholder
