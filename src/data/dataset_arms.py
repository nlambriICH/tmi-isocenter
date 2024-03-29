"""Dataset utility functions"""
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.dataset import Dataset


class DatasetArms(Dataset):
    """
    A dataset class for loading and stratifying data for patients with isocenters on the arms.

    Attributes:
        - df_patient_info (pd.DataFrame): DataFrame containing patient information.

    Methods:
        - train_val_test_split(): Split the dataset into train, validation, and test sets in a stratified manner.
        - unique_output(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat): Create the target array with
          minimum elements (only unique information).
    """

    def __init__(self) -> None:
        """
        Initialize the `DatasetArms` class.

        This constructor initializes the class and filters the dataset to work only with entries where isocenters
        are positioned on the arms.
        """
        super().__init__()
        iso_on_arms = self.df_patient_info.IsocenterOnArms.to_numpy(dtype=bool)
        self.df_patient_info = self.df_patient_info.iloc[iso_on_arms]

    def train_val_test_split(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into training, validation, and test sets.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray): Index splits for train, validation, and test sets.

        Notes:
            - The dataset split is stratified by the class labels (collimator angle).
            - The default split ratio is 80% for training, 10% for validation, and 10% for testing.
        """

        _, test_idx = train_test_split(
            self.df_patient_info.index,
            train_size=0.91,
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
                The resulting array has a shape of (self.num_patients, 1, 32).

        Notes:
            - The resulting array contains 7 values for the isocenters,
            18 values for the X_jaws, and 9 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, 30), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            # Isocenters
            unique_iso_idx = [
                30,
                33,
            ]  # indexes: 30,33 = two different X-coord on the arms
            y_iso_new2 = np.zeros(shape=(5), dtype=float)
            y_iso_new1 = iso[unique_iso_idx]
            for z in range(2):
                y_iso_new2[z] = iso[
                    z * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso
            # Skip the third iso.
            for z in range(3):
                y_iso_new2[z + 2] = iso[
                    (z + 3) * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso

            # X_Jaws: take all the values except the for thorax, chest and head
            unused_idx = [
                5,  # overlap fourth iso with chest's field
                8,  # Third iso deleted
                9,  # Third iso deleted
                10,  # Third iso deleted
                11,  # Third iso deleted
                13,  # overlap chest iso with head 's field
                15,  # chest symmetry on iso
                19,  # head symmetry on iso
            ]
            y_jaw_X = np.delete(jaw_X_pix, unused_idx)

            # Y_Jaws: exploit the body's symmetry
            unique_Y_idx = [
                0,
                2,
                4,
                16,
                17,
                18,
                19,
            ]
            # Keep [0,2] legs, 4 = one values fields (pelvis + chest), 8 = third iso, [16,17,18,19] head, [20,22] = arms
            y_jaw_Y = jaw_Y_pix[unique_Y_idx]
            y_reg_local = np.concatenate(
                (y_iso_new1, y_iso_new2, y_jaw_X, y_jaw_Y), axis=0
            )
            y_reg[i] = y_reg_local

        return y_reg
