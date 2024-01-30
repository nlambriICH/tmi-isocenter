"""Lightning module for CNN training"""
import torch
import torch.nn.functional as F

from src.config.constants import OUTPUT_DIM
from src.modules.lightning_cnn import LitCNN


class BodyCNN(LitCNN):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        weight=3,
        focus_on=[1],
        filters=4,
        output=OUTPUT_DIM,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
        """
        super().__init__(
            learning_rate=learning_rate,
            mse_loss_weight=mse_loss_weight,
            weight=weight,
            focus_on=focus_on,
            filters=filters,
            output=output,
        )

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

        x, y_reg = batch
        y_reg = y_reg.view(y_reg.size(0), -1)  # shape=(N_batch, N_out)

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
        x, y_reg = batch
        y_reg = y_reg.view(1, -1)  # shape=(1, N_out)

        y_reg_hat = self.cnn(x)
        val_mse_loss = self.weighted_mse_loss(y_reg_hat, y_reg)
        metrics = {
            "val_mse_loss": val_mse_loss,
        }
        self.log_dict(metrics)
