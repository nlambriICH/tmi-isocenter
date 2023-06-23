"""Lightning module for CNN training"""
import os
from src.modules.lightning_cnn import LitCNN
import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy
from src.models.cnn import CNN
from src.config.constants import CLASSIFICATION


class ArmCNN(LitCNN):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        bcelogits_loss_weight=0.0000001,
        weight=1,
        activation=nn.ReLU(),
        focus_on=[0, 1],
        filters=4,
        output=32,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
                and angle classification
        """
        super().__init__()
        self.example_input_array = torch.Tensor(
            32, 3, 512, 512
        )  # display the intermediate input and output sizes of layers when trainer.fit() is called
        self.flag = (CLASSIFICATION,)
        self.cnn = CNN(filters, output, self.flag, activation)
        self.accuracy = BinaryAccuracy()
        self.learning_rate = learning_rate
        self.train_mse_weight = mse_loss_weight
        self.bcelogits_loss_weight = bcelogits_loss_weight
        self.weights = torch.ones(output)
        self.weights[focus_on] = weight
        self.save_hyperparameters()
        self.filters = filters
