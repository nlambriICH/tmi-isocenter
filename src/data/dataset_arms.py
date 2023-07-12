"""Dataset utility functions"""
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.dataset import Dataset


class DatasetArms(Dataset):
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        super().__init__()
        iso_on_arms = self.df_patient_info.IsocenterOnArms.to_numpy(dtype=bool)
        self.df_patient_info = self.df_patient_info.iloc[iso_on_arms]

    def train_val_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the train/val/test indexes :
            - 10% patients for test set
            - 80% for training
            - 10% for validation

        Args:
            test_set (str): The strategy to split the test data based on the  class labels (collimator angle).


        Returns:
            tuple(np.ndarray, np.ndarray): train, val, and test index splits

        Notes:
            The dataset split is stratified by the class labels (collimator angle)
        """

        _, test_idx = train_test_split(
            self.df_patient_info.index,
            train_size=0.91,
        )  # get index as a balance test_set
        test_idx = test_idx.to_numpy(dtype=np.uint8)
        train_idx = self.df_patient_info.index[
            ~self.df_patient_info.index.isin(test_idx)  # remove test_idx from dataframe
        ].to_numpy(dtype=np.uint8)

        train_idx, val_idx = train_test_split(
            train_idx,
            train_size=0.9,
        )
        self.train_idx = train_idx
        return (train_idx, val_idx, test_idx)

    def unique_output(
        self, isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
    ) -> np.ndarray:
        """
        Create a new data configuration for the input of the model with iso on the arms.

        Args:
            isocenters_pix_flat (np.ndarray): Flat array containing the isocenter values.
            jaws_X_pix_flat (np.ndarray): Flat array containing the X_jaws values.
            jaws_Y_pix_flat (np.ndarray): Flat array containing the Y_jaws values.

        Returns:
            np.ndarray: Array with the unique values from the input data.
                The resulting array has a shape of (self.num_patients, 1, 32).

        Notes:
            - The resulting array contains 7 values for the isocenters,
            18 values for the X_jaws, and 9 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, 32), dtype=float)
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
                5,  # overlap fourth iso with chest's field
                8,  # Third iso deleted
                9,  # Third iso deleted
                10,  # Third iso deleted
                11,  # Third iso deleted
                13,  # overlap chest iso with head 's field
                15,  # chest symmetry on iso
                19,  # head symmetry on iso
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
