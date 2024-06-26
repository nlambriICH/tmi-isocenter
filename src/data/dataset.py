"""Dataset utility functions"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from src.config.constants import INTERIM_DATA_DIR_PATH
from src.data.augmentation import Augmentation


class Dataset:
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        self.masks2d = np.transpose(
            np.load(join(INTERIM_DATA_DIR_PATH, "masks2D.npy")), (0, 3, 1, 2)
        )
        self.normalize_ptv_hu()
        self.num_patients = self.masks2d.shape[0]

        self.isocenters_pix = np.load(
            join(INTERIM_DATA_DIR_PATH, "isocenters_pix.npy")
        )  # shape=(N, 12, 3)

        self.jaws_X_pix = np.load(
            join(INTERIM_DATA_DIR_PATH, "jaws_X_pix.npy")
        )  # shape=(N, 12, 2)
        self.jaws_Y_pix = np.load(
            join(INTERIM_DATA_DIR_PATH, "jaws_Y_pix.npy")
        )  # shape=(N, 12, 2)
        self.angles = np.load(
            join(INTERIM_DATA_DIR_PATH, "angles.npy")
        )  # shape=(N, 12)
        self.angle_class = np.where(self.angles[:, 0] == 90, 0.0, 1.0)  # shape=(N,)
        self.df_patient_info_original = pd.read_csv(
            join(dirname(INTERIM_DATA_DIR_PATH), "patient_info.csv")
        )
        self.df_patient_info = self.df_patient_info_original

    def normalize_ptv_hu(self, background=0) -> None:
        """Normalize the channel corresponding to the PTV HU density.

        Parameters:
            self (object): The instance of the class containing the masks.
            background (int): Background value used for normalization. Default is -1.

        Returns:
            None

        Description:
            This function normalizes the HU mask of the PTV by applying a min-max normalization.
            The normalization is performed independently on each mask, ensuring
            that the minimum value of each mask becomes 0 and the maximum value becomes 1.
            If the `background` parameter is provided, the normalization is performed by considering
            only the non-zero values within each mask, with the background value specified.
            Otherwise, if `background` is set to -1, the entire mask range is considered for normalization.
            The resulting normalized masks are stored in a new numpy array `norm_ptv` which is returned.
        """
        for i, mask2 in enumerate(self.masks2d):
            mask_hu = mask2[0]
            mask_ptv = mask2[1]
            non_zero_values = mask_hu[np.nonzero(mask_ptv)]
            min_value = np.min(non_zero_values) if background == 0 else np.min(mask_hu)
            max_value = np.max(non_zero_values) if background == 0 else np.max(mask_hu)
            difference = max_value - min_value
            normalized = (
                np.where(mask_ptv != 0, (mask_hu - min_value) / difference, background)
                if background == 0
                else (mask_hu - min_value) / difference
            )

            self.masks2d[i, 0] = normalized

    def train_val_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): Index splits for train, validation, and test sets.

        Notes:
            - The default split ratio is 80% for training, 10% for validation, and 10% for testing.
        """

        _, test_idx = train_test_split(
            self.df_patient_info.index,
            train_size=0.91,
            random_state=42,
        )
        test_idx = test_idx.to_numpy()
        train_idx = self.df_patient_info.index[
            ~self.df_patient_info.index.isin(
                test_idx
            )  # remove test_idx from data frame
        ].to_numpy()

        train_idx, val_idx = train_test_split(
            train_idx,
            train_size=0.9,
        )

        self.train_idx = train_idx
        return (train_idx, val_idx, test_idx)

    def augment_train(self):
        """
        Augment training data.

        """
        aug = Augmentation(
            self.masks2d,
            self.train_idx,
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
            self.df_patient_info_original,
        )

        (
            self.masks2d,
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angle_class,
            self.df_patient_info,
            train_index,
        ) = aug.augment()

        self.num_patients = self.masks2d.shape[0]

        return train_index

    def get_dataset(self) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        """
        Prepares and returns the three datasets needed for training a model from:
            - Input data (X): Array of images of shape (C, H, W).
            - Target data (y_reg): Array of unique outputs derived from isocenters and jaws positions.


        Returns:
            tuple[TensorDataset, TensorDataset, TensorDataset]: A tuple containing:
                - Train dataset: TensorDataset for training of the model.
                - Validation dataset: TensorDataset for validation of the model.
                - Test dataset: TensorDataset for testing of the model.
        """
        isocenters_pix_flat = self.isocenters_pix.reshape(self.num_patients, -1)
        jaws_X_pix_flat = self.jaws_X_pix.reshape(self.num_patients, -1)
        jaws_Y_pix_flat = self.jaws_Y_pix.reshape(self.num_patients, -1)
        y_reg = self.unique_output(
            isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
        )

        _, val_idx, test_index = self.train_val_test_split()
        train_index = self.augment_train()

        (
            masks_train,
            y_reg_train,
            masks_val,
            y_reg_val,
            masks_test,
            y_reg_test,
            test_idx,
            train_index,
        ) = tuple(
            map(
                torch.Tensor,
                (
                    self.masks2d[train_index],
                    y_reg[train_index],
                    self.masks2d[val_idx],
                    y_reg[val_idx],
                    self.masks2d[test_index],
                    y_reg[test_index],
                    test_index,
                    train_index,
                ),
            )
        )

        train_dataset = TensorDataset(
            masks_train,
            y_reg_train,
        )

        val_dataset = TensorDataset(
            masks_val,
            y_reg_val,
        )

        test_len = test_index.shape[0]
        test_dataset = TensorDataset(
            masks_test,
            y_reg_test,
            test_idx,
            masks_train[0:test_len],  # mask_train = [0:11] or [0:3]
            train_index[0:test_len],  # mask_train = [0:11] or [0:3]
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
        )

    def prediction_dataset(self) -> TensorDataset:
        """
        Prepares and returns a dataset to evaluate the model's predictions.

        Returns:
            TensorDataset: The TensorDataset containing the data to evaluate the model.
        """
        isocenters_pix_flat = self.isocenters_pix.reshape(self.num_patients, -1)
        jaws_X_pix_flat = self.jaws_X_pix.reshape(self.num_patients, -1)
        jaws_Y_pix_flat = self.jaws_Y_pix.reshape(self.num_patients, -1)
        y_reg = self.unique_output(
            isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
        )

        _, _, predict_index = self.train_val_test_split()

        (
            masks,
            y_reg,
            predict_index,
        ) = tuple(
            map(
                torch.Tensor,
                (
                    self.masks2d[predict_index],
                    y_reg[predict_index],
                    predict_index,
                ),
            )
        )

        pred_dataset = TensorDataset(
            masks,
            y_reg,
            predict_index,
        )

        return pred_dataset

    def unique_output(
        self, isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
    ) -> np.ndarray:
        """
        Create the target array with minimum elements (only unique information).

        Args:
            isocenters_pix_flat (np.ndarray): Flat array containing the isocenter values.
            jaws_X_pix_flat (np.ndarray): Flat array containing the X_jaws values.
            jaws_Y_pix_flat (np.ndarray): Flat array containing the Y_jaws values.

        Returns:
            np.ndarray: Array with the unique values from the input data.
                The resulting array has a shape of (self.num_patients, 1, 39).

        Notes:
            - The resulting array contains 8 values for the isocenters,
            21 values for the X_jaws, and 10 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, 39), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            # Isocenters
            unique_iso_idx = [
                30,
                33,
            ]  # indexes: 0,1 = one coord. for x,y axes; 30,33 = two different X-coord on the arms
            y_iso_new2 = np.zeros(shape=(6), dtype=float)
            y_iso_new1 = iso[unique_iso_idx]
            for z in range(6):
                y_iso_new2[z] = iso[
                    z * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso

            # X_Jaws: take all the values except the for thorax, chest and head
            unused_idx = [
                11,
                15,
                19,
            ]
            y_jaw_X = np.delete(jaw_X_pix, unused_idx)

            # Y_Jaws: exploit the body's symmetry
            unique_Y_idx = [
                0,
                2,
                4,
                8,
                16,
                17,
                18,
                19,
                20,
                22,
            ]
            # Keep [0,2] legs, 4 = one values fields (pelvis + chest), 8 = third iso, [16,17,18,19] head, [20,22] = arms
            y_jaw_Y = jaw_Y_pix[unique_Y_idx]
            y_reg_local = np.concatenate(
                (y_iso_new1, y_iso_new2, y_jaw_X, y_jaw_Y), axis=0
            )
            y_reg[i] = y_reg_local

        return y_reg

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
