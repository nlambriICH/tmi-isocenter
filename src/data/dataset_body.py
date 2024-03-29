"""Dataset utility functions"""
import numpy as np
from src.data.dataset import Dataset


class DatasetBody(Dataset):
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        super().__init__()
        iso_on_arms = self.df_patient_info.IsocenterOnArms.to_numpy(dtype=bool)
        self.df_patient_info = self.df_patient_info.iloc[~iso_on_arms]

    def unique_output(
        self, isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
    ) -> np.ndarray:
        """
        Create a new data configuration for the input of the model with isocenter only on the body.

        Args:
            isocenters_pix_flat (np.ndarray): Flat array containing the isocenter values.
            jaws_X_pix_flat (np.ndarray): Flat array containing the X_jaws values.
            jaws_Y_pix_flat (np.ndarray): Flat array containing the Y_jaws values.

        Returns:
            np.ndarray: Array with the unique values from the input data.
                The resulting array has a shape of (self.num_patients, 1, 34).

        Notes:
            - The resulting array contains 6 values for the isocenters,
            17 values for the X_jaws, and 7 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, 25), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            # Isocenters
            y_iso_new2 = np.zeros(shape=(4), dtype=float)
            for z in range(2):
                y_iso_new2[z] = iso[
                    z * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso
                y_iso_new2[z + 2] = iso[
                    (z + 3) * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso

            # X_Jaws: take all the values except the for thorax, chest and head
            unused_idx = [
                5,  # overlap fourth
                9,  # overlap third
                11,  # third on iso
                13,  # overlap chest
                15,  # chest symmetry on iso
                19,  # head symmetry on iso
                20,  # arms
                21,  # arms
                22,  # arms
                23,  # arms
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
            y_reg_local = np.concatenate((y_iso_new2, y_jaw_X, y_jaw_Y), axis=0)
            y_reg[i] = y_reg_local

        return y_reg
