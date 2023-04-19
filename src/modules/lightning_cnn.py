"""Lightning module for CNN training"""
import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from torchmetrics.classification import BinaryAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from src.models.cnn import CNN


class LitCNN(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        bcelogits_loss_weight=1.0,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
                and angle classification
        """
        super().__init__()
        self.example_input_array = torch.Tensor(
            32, 1, 512, 512
        )  # display the intermediate input and output sizes of layers when trainer.fit() is called
        self.cnn = CNN()
        self.accuracy = BinaryAccuracy()
        self.learning_rate = learning_rate
        self.train_mse_weight = mse_loss_weight
        self.bcelogits_loss_weight = bcelogits_loss_weight
        self.save_hyperparameters()

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
        x, y_reg, y_cls = batch
        x = x.unsqueeze_(1)  # shape=(N_batch, 1, 512, 512)
        y_reg = y_reg.view(y_reg.size(0), -1)  # shape=(N_batch, N_out)
        y_cls = y_cls.view(-1, 1)  # shape=(N_batch, 1)
        y_reg_hat, y_cls_hat = self.cnn(x)
        train_mse_loss = F.mse_loss(y_reg_hat, y_reg)
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
        # need additional (batch) dimension because Flatten layer has start_dim=1
        x = x.unsqueeze_(0)  # shape=(1, 1, 512, 512)
        y_reg = y_reg.view(1, -1)  # shape=(1, N_out)
        y_cls = y_cls.view(1, -1)  # shape=(1, 1)
        y_reg_hat, y_cls_hat = self.cnn(x)
        val_mse_loss = F.mse_loss(y_reg_hat, y_reg)
        self.accuracy(torch.sigmoid(y_cls_hat), y_cls)
        metrics = {"val_mse_loss": val_mse_loss, "val_accuracy": self.accuracy}
        self.log_dict(metrics)

    def test_step(  # pylint: disable=arguments-differ
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> None:
        """Test loop

        Args:
            batch (list[torch.Tensor]): input batch
            batch_idx (int): batch index
        """
        x, y_reg, y_cls, patient_id = batch
        # need additional (batch) dimension because Flatten layer has start_dim=1
        x = x.unsqueeze_(0)  # shape=(1, 1, 512, 512)
        y_reg = y_reg.view(1, -1)  # shape=(1, N_out)
        y_cls = y_cls.view(1, -1)  # shape=(1, N_out)
        y_reg_hat, y_cls_hat = self.cnn(x)
        test_mse_loss = F.mse_loss(y_reg_hat, y_reg)
        self.accuracy(torch.sigmoid(y_cls_hat), y_cls)
        metrics = {"test_mse_loss": test_mse_loss, "test_accuracy": self.accuracy}
        self.log_dict(metrics)
        # TODO: add plot predictions

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
