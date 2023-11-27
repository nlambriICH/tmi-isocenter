"""Lightning module for CNN training"""
import torch
import torch.nn.functional as F
from src.modules.lightning_cnn import LitCNN


class BodyCNN(LitCNN):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        bcelogits_loss_weight=0.00000001,
        weight=3,
        focus_on=[1],
        filters=4,
        output=25,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
                and angle classification
        """
        super().__init__(
            learning_rate=learning_rate,
            mse_loss_weight=mse_loss_weight,
            bcelogits_loss_weight=bcelogits_loss_weight,
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
