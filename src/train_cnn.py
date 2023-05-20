"""Script for model training"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from src.data.dataset import Dataset
from src.modules.lightning_cnn import LitCNN


if __name__ == "__main__":
    dataset = Dataset()
    train_index, val_idx, test_index = dataset.train_val_test_split(test_set="balanced")
    masks_aug, y_reg, y_cls = dataset.get_data_Xy()

    (
        masks_train,
        y_reg_train,
        y_cls_train,
        masks_val,
        y_reg_val,
        y_cls_val,
        masks_test,
        y_reg_test,
        y_cls_test,
        test_idx,
        train_index,
    ) = tuple(
        map(
            torch.Tensor,
            (
                masks_aug[train_index],
                y_reg[train_index],
                y_cls[train_index],
                masks_aug[val_idx],
                y_reg[val_idx],
                y_cls[val_idx],
                masks_aug[test_index],
                y_reg[test_index],
                y_cls[test_index],
                test_index,
                train_index,
            ),
        )
    )

    train_loader = DataLoader(
        TensorDataset(masks_train, y_reg_train, y_cls_train, train_index),
        num_workers=1,
        batch_size=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(masks_val, y_reg_val, y_cls_val),
        num_workers=1,
    )
    test_loader = DataLoader(
        TensorDataset(masks_test, y_reg_test, y_cls_test, test_idx),
        num_workers=1,
    )

    lightning_cnn = LitCNN()
    trainer = pl.Trainer(
        callbacks=[  # pyright: ignore[reportGeneralTypeIssues]
            EarlyStopping(monitor="val_mse_loss", mode="min", patience=10),
            ModelSummary(
                max_depth=-1
            ),  # print the weights summary of the model when trainer.fit() is called
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=40,
        log_every_n_steps=1,
    )
    trainer.fit(
        model=lightning_cnn, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(model=lightning_cnn, dataloaders=test_loader)
