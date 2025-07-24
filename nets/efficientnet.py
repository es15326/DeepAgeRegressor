# efficientnet_model.py

import torch
import torch.nn as nn
import torchvision.models as models
from .base import BaseNet
# Assuming your custom loss function is in a file named loss.py
# from loss import weighted_mse_loss

class EfficientNetAgeModel(BaseNet):
    """
    An age estimation model using the EfficientNetV2-S architecture.
    """
    def __init__(self, pretrained=True):
        super().__init__() #

        # 1. Load a powerful, modern, pretrained model
        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.efficientnet_v2_s(weights=weights)

        # 2. Replace the final layer for the regression task
        # Get the number of input features for the classifier
        in_features = self.model.classifier[1].in_features
        # The new output layer predicts a single value (age)
        self.model.classifier = nn.Linear(in_features, 1)

        # 3. Set the loss function (can be your custom one)
        # self.loss_func = weighted_mse_loss # As in the original file
        self.loss_func = nn.MSELoss() # Using a standard loss as a placeholder
