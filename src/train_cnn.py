"""Script for model training"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.config.constants import MODEL

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

    train_index, val_idx, test_index = dataset.train_val_test_split()
    # test_index=np.array([59,60,61,62,63,64,65,66],) # Custom list of patients to use as test set for the model.
    train_index = dataset.augment_train()
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
        shuffle=True,
        num_workers=3,
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

    # Uncomment this part ito evaluate a patient saved in a precedent model!
    # my_model=lightning_cnn
    # tersn =  my_model.load_from_checkpoint(r"D:\tmi-isocenter-1\lightning_logs\body_model\version_118\checkpoints\epoch=22-step=414.ckpt") ""example_of_path""
    # trainer.test(model=tersn, dataloaders=test_loader)

    trainer.fit(
        model=lightning_cnn, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(model=lightning_cnn, dataloaders=test_loader)
