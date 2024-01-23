"""Convolutional neural network module"""
import torch
from torch import nn


class CNN(nn.Module):
    """Simple CNN for model testing"""

    def __init__(
        self,
        filters: int,
        output: int,
    ):
        super().__init__()

        self.simple_cnn = nn.Sequential(
            nn.Conv2d(
                3,
                filters,
                8,
            ),
            nn.ReLU(),
            nn.Conv2d(
                filters,
                filters * 2,
                8,
            ),
            nn.ReLU(),
            nn.Conv2d(
                filters * 2,
                filters * 4,
                8,
            ),
            nn.ReLU(),
            nn.Conv2d(
                filters * 4,
                filters * 8,
                8,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters * 8 * 242 * 242, output),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network

        Args:
            x (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output tensor for regression.
        """
        x = self.simple_cnn(x)

        return self.regression_head(x)
