"""Dataset utility functions"""
from typing import Literal
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class Dataset:
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        with np.load(r"data\interim\masks2D.npz") as npz_masks2d:  # shape=(N, 512, z)
            try:
                self.masks2d = np.array(list(npz_masks2d.values()))
            except ValueError:
                traceback.print_exc()
                print(
                    "Could not create NumPy array of masks. Please ensure they are square matrices."
                )

        self.num_patients = self.masks2d.shape[0]

        self.isocenters_pix = np.load(
            r"data\interim\isocenters_pix.npy"
        )  # shape=(N, 12, 3)

        self.jaws_X_pix = np.load(r"data\interim\jaws_X_pix.npy")  # shape=(N, 12, 2)
        self.jaws_Y_pix = np.load(r"data\interim\jaws_Y_pix.npy")  # shape=(N, 12, 2)

        self.angles = np.load(r"data\interim\angles.npy")  # shape=(N, 12)
        self.angle_class = np.where(self.angles[:, 0] == 90, 0.0, 1.0)  # shape=(N,)

        self.df_patient_info = pd.read_csv(r"data\patient_info.csv").sort_values(
            by="PlanDate"
        )

    def train_val_test_split(
        self, test_set: Literal["date", "oldest", "latest"]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the train/val/test indexes :
            - 10 patients for test set
            - 90 for training
            - 10 for validation

        Args:
            test_set (str): The strategy to split the test data based on the treatment date.
            Either ordered by date, the 10 oldest, or 10 latest.

        Returns:
            tuple(np.ndarray, np.ndarray): train, val, and test index splits

        Notes:
            The dataset split is not stratified by the class labels (collimator angle)
        """

        if test_set == "date":
            test_idx = self.df_patient_info.iloc[10::10].index.to_numpy(
                dtype=np.uint8
            )  # get index of every 10th row
        elif test_set == "oldest":
            test_idx = self.df_patient_info.iloc[:10].index.to_numpy(
                dtype=np.uint8
            )  # get index of the first 10th rows
        elif test_set == "latest":
            test_idx = self.df_patient_info.iloc[-10:].index.to_numpy(
                dtype=np.uint8
            )  # get index of the last 10th rows
        else:
            raise ValueError(
                f'test_set must be "data" or "oldest" or "latest" but was {test_set}'
            )

        train_idx = self.df_patient_info.index[
            ~self.df_patient_info.index.isin(test_idx)
        ].to_numpy(
            dtype=np.uint8
        )  # remove test_idx from dataframe

        train_idx, val_idx = next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.1).split(
                np.zeros_like(train_idx), self.angle_class[train_idx - 1]
            )
        )

        imb_ratio_train = np.sum(self.angle_class[train_idx] == 0) / np.sum(
            self.angle_class[train_idx] == 1
        )
        imb_ratio_val = np.sum(self.angle_class[val_idx] == 0) / np.sum(
            self.angle_class[val_idx] == 1
        )
        imb_ratio_test = np.sum(self.angle_class[test_idx] == 0) / np.sum(
            self.angle_class[test_idx] == 1
        )
        print(f"Imbalance ratio train set: {imb_ratio_train:.1f}")
        print(f"Imbalance ratio val set: {imb_ratio_val:.1f}")
        print(f"Imbalance ratio test set: {imb_ratio_test:.1f}")

        return (train_idx, val_idx, test_idx)

    def get_data_Xy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        isocenters_pix_flat = self.isocenters_pix.reshape(self.num_patients, -1)
        jaws_X_pix_flat = self.jaws_X_pix.reshape(self.num_patients, -1)
        jaws_Y_pix_flat = self.jaws_Y_pix.reshape(self.num_patients, -1)

        return (
            self.masks2d,
            np.concatenate(
                (isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat), axis=1
            ),
            self.angle_class,
        )

    @property
    def get_patient_info(self) -> pd.DataFrame:
        return self.df_patient_info

    @property
    def get_masks(self) -> np.ndarray:
        return self.masks2d

    @property
    def get_isocenters_pix(self) -> np.ndarray:
        return self.isocenters_pix

    @property
    def get_jaws_X_pix(self) -> np.ndarray:
        return self.jaws_X_pix

    @property
    def get_jaws_Y_pix(self) -> np.ndarray:
        return self.jaws_Y_pix

    @property
    def get_angle_class(self) -> np.ndarray:
        return self.angle_class
