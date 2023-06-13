"""Script for model training"""
import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils.visualization_cnn import model


if __name__ == "__main__":
    if model == "arms":
        from src.data.dataset_arms import DatasetArms
        from src.modules.arms_cnn import ArmCNN

        dataset = DatasetArms()
        lightning_cnn = ArmCNN()
        name = "arms_model"
    elif model == "body":
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

    train_index, val_idx, test_index = dataset.train_val_test_split()
    masks_aug, y_reg, y_cls = dataset.get_data_Xy()
    logger = TensorBoardLogger(
        "lightning_logs",
        name=name,
        log_graph=True,
    )

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
    test_len = len(test_index)
    train_loader = DataLoader(
        TensorDataset(
            masks_train,
            y_reg_train,
            y_cls_train,
        ),
        num_workers=1,
        batch_size=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(masks_val, y_reg_val, y_cls_val),
        num_workers=1,
    )
    test_loader = DataLoader(
        TensorDataset(
            masks_test,
            y_reg_test,
            y_cls_test,
            test_idx,
            masks_train[
                0:test_len
            ],  # it switches if the dataset changes: [0:11] or [0:3]
            train_index[
                0:test_len
            ],  # it switches if the dataset changes: [0:11] or [0:3]
        ),
        num_workers=1,
    )

    trainer = pl.Trainer(
        logger=logger,  # pyright: ignore[reportGeneralTypeIssues]
        callbacks=[  # pyright: ignore[reportGeneralTypeIssues]
            EarlyStopping(monitor="val_mse_loss", mode="min", patience=7),
            ModelSummary(
                max_depth=-1
            ),  # print the weights summary of the model when trainer.fit() is called
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=100,
        log_every_n_steps=1,
    )
    trainer.fit(
        model=lightning_cnn, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(model=lightning_cnn, dataloaders=test_loader)
