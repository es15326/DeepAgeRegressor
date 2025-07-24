import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang


class WeightedRegressionLoss(nn.Module):
    def __init__(self, dataset: any):
        super().__init__()
        
        self.max_target = 81
        self.lds_ks = 5
        self.lds_sigma = 2
        self.lds_kernel = "gaussian"
        self.df = dataset if type(dataset) == pd.DataFrame else dataset.df
        
        self.weights = self._prepare_weights()
        
    def forward(self, inputs, targets):
        raise NotImplementedError
    
    def _prepare_weights(self):
        value_dict = {x: 0 for x in range(self.max_target + 1)}
        labels = self.df["label"].values
        for label in labels:
            value_dict[int(label)] += 1
        value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        lds_kernel = self._get_lds_kernel(self.lds_kernel, self.lds_ks, 
                                          self.lds_sigma)
        smoothed_value = convolve1d(
            np.asarray([v for _, v in value_dict.items()]),
            weights=lds_kernel,
            mode="constant",
        ) 
        weight_map = 1 / (smoothed_value + 10e-6)
        weight_map *= len(weight_map[labels]) / weight_map[labels].sum()
        print()
        print('----*')
        print("weight map: ", weight_map)
        print('----*')
        print()
        
        return torch.tensor(weight_map)
    
    def _get_lds_kernel(self, kernel, ks, sigma):
        assert kernel in ["gaussian", "triang", "laplace"]
        half_ks = (ks - 1) // 2
        if kernel == "gaussian":
            base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                gaussian_filter1d(base_kernel, sigma=sigma)
            )
        elif kernel == "triang":
            kernel_window = triang(ks)
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                map(laplace, np.arange(-half_ks, half_ks + 1))
            )

        return kernel_window

class WeightedMSELoss(WeightedRegressionLoss):
    def forward(self, inputs, targets):
        loss = (inputs - targets) ** 2
        device = targets.device
        weights = self.weights.to(device)[targets.long()]
        # weights = self.weights.to(device)[targets]
        # weights = self.weights[targets].to(loss.device)
        loss *= weights
        loss = torch.sum(loss) / torch.sum(weights)
        return loss

class WeightedL1Loss(WeightedRegressionLoss):
    def forward(self, inputs, targets):
        loss = torch.abs(inputs - targets)
        weights = self.weights[targets].to(loss.device)
        loss *= weights
        loss = torch.sum(loss) / torch.sum(weights)
        return loss
    
class WeightedAARLoss(WeightedRegressionLoss):
    def forward(self, inputs, targets):
        self.weights = self.weights.to(inputs.device)
        weights = self.weights[targets.long()]
        # weights = self.weights[targets]
        mae = F.smooth_l1_loss(inputs, targets.float(), reduce=False)
        mae *= weights
        mae = torch.sum(mae) / torch.sum(weights)
        
        std = 0
        true_age_groups = torch.clip(targets // 10, 0, 7)
        for i in range(8):
            idx = true_age_groups == i
            if targets[idx].shape[0] != 0:
                mae_age_group = F.smooth_l1_loss(inputs[idx], 
                                                 targets[idx].float(), 
                                                 reduce=False)
                mae_age_group *= weights[idx]
                mae_age_group = torch.sum(mae_age_group) / torch.sum(weights[idx])
                std += (mae_age_group - mae) ** 2

        return 7 * mae + 3 * torch.sqrt(std / 8)

      
def soft_aar_loss(y_pred: torch.FloatTensor, y_true: torch.Tensor):
    mae = F.smooth_l1_loss(y_pred, y_pred)
    true_age_groups = torch.clip(y_true // 10, 0, 7)
    std = 0
    for i in range(8):
        idx = true_age_groups == i
        if y_true[idx].shape[0] != 0:
            mae_age_group = F.smooth_l1_loss(y_true[idx] * 81, y_pred[idx] * 81)
            std += (mae_age_group - mae) ** 2

    return 7 * mae + 3 * torch.sqrt(std / 8)

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights
    loss = torch.mean(loss)
    return loss
