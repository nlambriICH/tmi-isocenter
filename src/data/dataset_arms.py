"""Dataset utility functions"""
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from src.data.dataset import Dataset


class Dataset_arms(Dataset):
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        super().__init__()

        # Creating the class interaction between arms and angle
        self.iso_on_arms = self.df_patient_info["IsocenterOnArms"].to_numpy()
        bool_arms = self.iso_on_arms.astype(bool)
        self.df_patient_info = self.df_patient_info[bool_arms].reset_index()

    def unique_output(
        self, isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
    ) -> np.ndarray:
        """
        Create a new data configuration for the input of the models.

        Args:
            isocenters_pix_flat (np.ndarray): Flat array containing the isocenter values.
            jaws_X_pix_flat (np.ndarray): Flat array containing the X_jaws values.
            jaws_Y_pix_flat (np.ndarray): Flat array containing the Y_jaws values.

        Returns:
            np.ndarray: Array with the unique values from the input data.
                The resulting array has a shape of (self.num_patients, 1, 34).

        Notes:
            - The resulting array contains 11 values for the isocenters,
            21 values for the X_jaws, and 10 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, 34), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            # Isocenters
            unique_iso_idx = [
                30,
                33,
            ]  # indexes: 30,33 = two different X-coord. on the arms;
            y_iso_new2 = np.zeros(shape=(5), dtype=float)
            y_iso_new1 = iso[unique_iso_idx]
            for z in range(2):
                y_iso_new2[z] = iso[
                    z * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso.
            # Skip the third iso.
            for z in range(3):
                y_iso_new2[z + 2] = iso[
                    (z + 3) * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso.
            # X_Jaws
            usless_idx = [
                8,
                9,
                10,
                11,
                15,
                19,
            ]  # with X_Jaw I take all the values except the for thorax, chest and head where I use the simmetry (so 1 param for 2 fields) to hug the relative iso.
            y_jaw_X = np.delete(jaw_X_pix, usless_idx)
            # Y_Jaws
            unique_Y_idx = [
                0,
                2,
                4,
                16,
                17,
                18,
                19,
                20,
                22,
            ]  # Here we exploit the body's symmetry.
            # We keep [0,2] for the legs,  4 = one values fields (pelvi+chest), 8= third iso, [16,17,18,19] for the head, [20,22]= for the arms.
            y_jaw_Y = jaw_Y_pix[unique_Y_idx]
            y_reg_local = np.concatenate(
                (y_iso_new1, y_iso_new2, y_jaw_X, y_jaw_Y), axis=0
            )
            y_reg[i] = y_reg_local
        return y_reg

    def train_val_test_split(
        self, test_set: Literal["date", "oldest", "latest", "balanced"]
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
            The dataset split is stratified by the class labels (collimator angle)
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
        elif test_set == "balanced":
            test_idx = train_test_split(
                self.df_patient_info.index,
                train_size=0.91,
            )[1].to_numpy(
                dtype=np.uint8
            )  # get index as a balance test_set
        else:
            raise ValueError(
                f'test_set must be "data" or "oldest" or "latest" but was {test_set}'
            )

        train_idx = self.df_patient_info.index[
            ~self.df_patient_info.index.isin(test_idx)
        ].to_numpy(
            dtype=np.uint8
        )  # remove test_idx from dataframe

        train_idx, val_idx = train_test_split(
            train_idx,
            train_size=0.9,
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
