"""Dataset utility functions"""
import numpy as np

from src.config.constants import COLL_5_355, OUTPUT_DIM
from src.data.dataset import Dataset


class DatasetBody(Dataset):
    """Dataset class to load and stratify data"""

    def __init__(self) -> None:
        """
        Initialize the `DatasetBody` class.

        This constructor initializes the class and filters the dataset to work only with entries
        without isocenters on the arms. Thus it changes the regression head's dimension depending on COLL_5_355.

        Notes:
            - The default output dimension is 25, which is the minimum number of parameters for the model with 90 degrees collimator angle.
            - For the the model with 5 and 355 degrees collimator angle the output dimension is 19.
        """

        self.output = OUTPUT_DIM

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
                The resulting array has a shape of (self.num_patients, 1, self.output).

        Notes:
            - The resulting array contains 6 values for the isocenters,
            17 values for the X_jaws, and 7 values for the Y_jaws.
            - Specific indices are used to select the unique values from the input arrays.
            Details about the selected indices can be found in the function implementation.
        """
        y_reg = np.zeros(shape=(self.num_patients, 1, self.output), dtype=float)
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

            unique_X_idx = [
                0,
                1,
                2,
                3,
                4,
                6,
                7,
                8,
                10,
                12,
                14,
                16,
                17,
                18,
            ]

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

            if (
                COLL_5_355
            ):  # additional unused Jaws' values due to leg fields symmetries
                for z in range(4):
                    if z < 2:
                        unique_Y_idx.remove(
                            z * 2
                        )  # remove [0,2] legs due to Y_Jaws being fixed
                    unique_X_idx.remove(
                        z
                    )  # remove [0,1,2,3] legs due to X_Jaws being symmetryc

            # Keep [0,1,2,3] legs, [4,6,7] Pelvis, [8,10] = third iso, [12,14] chest, [16,17,18] head
            y_jaw_X = jaw_X_pix[unique_X_idx]

            # Keep [0,2] legs, 4 = one values fields (pelvis + chest), 8 = third iso, [16,17,18,19] head
            y_jaw_Y = jaw_Y_pix[unique_Y_idx]
            y_reg_local = np.concatenate((y_iso_new2, y_jaw_X, y_jaw_Y), axis=0)
            y_reg[i] = y_reg_local

        return y_reg
