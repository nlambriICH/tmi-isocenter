"""Convolutional neural network module"""
from torch import nn
import torch


class CNN(nn.Module):
    """Simple CNN for model testing"""

    def __init__(self, filters: int, activation=nn.ReLU()):
        super().__init__()
        self.simple_cnn = nn.Sequential(
            nn.Conv2d(
                3,
                filters,
                8,
            ),
            activation,
            nn.Conv2d(
                filters,
                filters,
                8,
            ),
            activation,
            nn.MaxPool2d(2),
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 249 * 249, 39),
        )
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 249 * 249, 1),
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
