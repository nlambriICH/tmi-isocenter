"""Script for model training"""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.config.constants import MODEL, NUM_WORKERS

if __name__ == "__main__":
    if MODEL == "arms":
        from src.data.dataset_arms import DatasetArms
        from src.modules.arms_cnn import ArmCNN

        dataset = DatasetArms()
        lightning_cnn = ArmCNN()
        name = "arms_model"
    elif MODEL == "body":
        from src.data.dataset_body import DatasetBody
        from src.modules.body_cnn import BodyCNN

        dataset = DatasetBody()
        lightning_cnn = BodyCNN()
        name = "body_model"
    else:
        from src.data.dataset import Dataset
        from src.modules.lightning_cnn import LitCNN

        dataset = Dataset()
        lightning_cnn = LitCNN()
        name = "whole_model"

    train_dataset, val_dataset, test_dataset = dataset.get_dataset()

    train_loader = DataLoader(
        train_dataset,
        num_workers=NUM_WORKERS,
        batch_size=10,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=NUM_WORKERS,
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(
            "lightning_logs",
            name=name,
            log_graph=True,
        ),  # pyright: ignore[reportGeneralTypeIssues]
        callbacks=[  # pyright: ignore[reportGeneralTypeIssues]
            EarlyStopping(monitor="val_mse_loss", mode="min", patience=7),
            ModelSummary(
                max_depth=-1
            ),  # print the weights summary of the model when trainer.fit() is called
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=50,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=lightning_cnn, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    trainer.test(model=lightning_cnn, dataloaders=test_loader)
