"""Lightning module for CNN training"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy
from src.models.cnn import CNN
from src.utils.visualization_cnn import Visualize
from src.config.constants import CLASSIFICATION


class LitCNN(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        bcelogits_loss_weight=0.00000001,
        weight=1,
        focus_on=[0, 1],
        filters=4,
        output=39,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
                and angle classification
        """
        super().__init__()
        self.version = 0
        self.example_input_array = torch.Tensor(
            1, 3, 512, 512
        )  # display the intermediate input and output sizes of layers when trainer.fit() is called
        self.classif = CLASSIFICATION
        self.cnn = CNN(
            filters,
            output,
            self.classif,
        )
        self.accuracy = BinaryAccuracy()
        self.learning_rate = learning_rate
        self.train_mse_weight = mse_loss_weight
        self.bcelogits_loss_weight = bcelogits_loss_weight
        self.weights = torch.ones(output)
        self.weights[focus_on] = weight
        self.save_hyperparameters()
        self.filters = filters

    def weighted_mse_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the weighted mean squared error (MSE) loss between the inputs and targets.

        Args:
            inputs (torch.Tensor): The predicted inputs tensor.
            targets (torch.Tensor): The target tensor with the ground truth values.

        Returns:
            torch.Tensor: The weighted MSE loss.
        """
        return (self.weights * F.mse_loss(inputs, targets, reduction="none")).mean()

    def training_step(  # pylint: disable=arguments-differ
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training loop

        Args:
            batch (list[torch.Tensor]): input batch
            batch_idx (int): batch index

        Returns:
            torch.Tensor: loss value
        """
        if self.current_epoch == 1:
            self.logger.log_graph(self)  # pyright: ignore[reportOptionalMemberAccess]

        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(  # pyright: ignore[reportOptionalMemberAccess , reportGeneralTypeIssues]
                name, param, global_step=self.global_step
            )

        x, y_reg, y_cls = batch
        y_reg = y_reg.view(y_reg.size(0), -1)  # shape=(N_batch, N_out)
        y_cls = y_cls.view(-1, 1)  # shape=(N_batch, 1)
        if not self.classif:
            y_reg_hat, y_cls_hat = self.cnn(x)
            train_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            train_bcelogits_loss = F.binary_cross_entropy_with_logits(y_cls_hat, y_cls)
            train_loss = (
                self.train_mse_weight * train_mse_loss
                + self.bcelogits_loss_weight * train_bcelogits_loss
            )
            self.accuracy(torch.sigmoid(y_cls_hat), y_cls)
            metrics = {
                "train_mse_loss": train_mse_loss,
                "train_bcelogits_loss": train_bcelogits_loss,
                "train_loss": train_loss,
                "train_accuracy": self.accuracy,
            }
        else:
            y_reg_hat = self.cnn(x)
            train_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            train_loss = self.train_mse_weight * train_mse_loss
            metrics = {
                "train_mse_loss": train_mse_loss,
                "train_loss": train_loss,
            }
        self.log_dict(metrics)

        return train_loss

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> None:
        """Validation loop

        Args:
            batch (list[torch.Tensor]): input batch
            batch_idx (int): batch index
        """
        x, y_reg, y_cls = batch
        y_reg = y_reg.view(1, -1)  # shape=(1, N_out)
        y_cls = y_cls.view(1, -1)  # shape=(1, 1)

        if not self.classif:
            y_reg_hat, y_cls_hat = self.cnn(x)
            val_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            self.accuracy(torch.sigmoid(y_cls_hat), y_cls)
            metrics = {"val_mse_loss": val_mse_loss, "val_accuracy": self.accuracy}
        else:
            y_reg_hat = self.cnn(x)
            val_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            metrics = {
                "val_mse_loss": val_mse_loss,
            }
        self.log_dict(metrics)

    def test_step(  # pylint: disable=arguments-differ
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Test loop

        Args:
            batch (list[torch.Tensor]): input batch
            batch_idx (int): batch index
        """
        (
            x,
            y_reg,
            y_cls,
            test_idx,
            x_train,
            train_index,
        ) = batch

        y_reg = y_reg.view(1, -1)  # shape=(1, N_out)
        y_cls = y_cls.view(1, -1)  # shape=(1, N_out)

        if not self.classif:
            y_reg_hat, y_cls_hat = self.cnn(x)
            test_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            self.accuracy(torch.sigmoid(y_cls_hat), y_cls)
            metrics = {"test_mse_loss": test_mse_loss, "test_accuracy": self.accuracy}
            y_train_reg_hat, y_train_cls_hat = self.cnn(x_train)
        else:
            y_reg_hat = self.cnn(x)
            test_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
            metrics = {
                "test_mse_loss": test_mse_loss,
            }
            y_cls_hat = torch.zeros(1)
            y_train_cls_hat = torch.zeros(1)
        y_train_reg_hat = self.cnn(x_train)
        self.log_dict(metrics)
        # check overfitting
        path = os.path.join(
            self.logger.log_dir,  # pyright: ignore[reportGeneralTypeIssues, reportOptionalMemberAccess]
            "train_img",  # pyright: ignore[reportGeneralTypeIssues]
        )
        viz = Visualize(self.logger.log_dir)
        vis_image_train = x_train.numpy()[0, :, :, :]
        vis_image_test = x.numpy()[0, :, :, :]
        # Two plots, first one for the train and the second for the test images
        viz.plot_img(
            vis_image_train,
            patient_idx=int(train_index.item()),
            output=y_train_reg_hat[0],
            path=path,
            coll_angle_hat=y_cls_hat,
            single_fig=True,
        )
        viz.plot_img(
            vis_image_test,
            patient_idx=int(test_idx.item()),
            output=y_reg_hat[0],
            path=self.logger.log_dir,  # pyright: ignore[reportGeneralTypeIssues,reportOptionalMemberAccess]
            coll_angle_hat=y_train_cls_hat,
            mse=test_mse_loss,
        )

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.cnn(x)

    def configure_optimizers(self) -> dict:
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }
