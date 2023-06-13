"""Dataset utility functions"""
import numpy as np
from src.data.dataset import Dataset


class DatasetBody(Dataset):
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        super().__init__()
        iso_on_arms = self.df_patient_info.IsocenterOnArms.to_numpy(dtype=bool)
        self.df_patient_info = self.df_patient_info.iloc[~iso_on_arms].reset_index()

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
        y_reg = np.zeros(shape=(self.num_patients, 1, 30), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            # Isocenters

            y_iso_new2 = np.zeros(shape=(6), dtype=float)
            for z in range(6):
                y_iso_new2[z] = iso[
                    z * 3 * 2 + 2
                ]  # Z-coord one for every couple of iso.
            # X_Jaws
            usless_idx = [
                11,
                15,
                19,
                20,
                21,
                22,
                23,
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
            ]  # Here we exploit the body's symmetry.
            # We keep [0,2] for the legs,  4 = one values fields (pelvi+chest), 8= third iso, [16,17,18,19] for the head, [20,22]= for the arms.
            y_jaw_Y = jaw_Y_pix[unique_Y_idx]
            y_reg_local = np.concatenate((y_iso_new2, y_jaw_X, y_jaw_Y), axis=0)
            y_reg[i] = y_reg_local
        return y_reg
