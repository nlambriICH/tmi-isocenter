"""Convolutional neural network module"""
from torch import nn
import torch


class CNN(nn.Module):
    """Simple CNN for model testing"""

    def __init__(self):
        super().__init__()
        self.simple_cnn = nn.Sequential(
            nn.Conv2d(1, 20, 8),
            nn.ReLU(),
            nn.Conv2d(20, 20, 8),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 249 * 249, 84),
        )
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 249 * 249, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output tensors for regression
                and classification heads
        """
        x = self.simple_cnn(x)
        return self.regression_head(x), self.classification_head(x)
