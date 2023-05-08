"""Dataset utility functions"""
from typing import Literal
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
                stratify=self.angle_class[self.df_patient_info.index],
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
            train_idx, train_size=0.9, stratify=self.angle_class[train_idx]
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
        y_reg = self.unique_output(
            isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat
        )

        return (
            self.masks2d,
            y_reg,
            self.angle_class,
        )

    def unique_output(self, isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat):
        """
        y_reg= MI CREO IL VETTORE DI DATI CORRETTO, QUELLI CHE MI SERVONO
         ISO   [X8, Y8, Xnull, Ynull,+Xbraccia,+Ybraccia,Xbraccia, Z1, Z2, Z3, Z4....Z6, ...   TOT ISO=13
        # CampiX [QUASI TUTTI] Nei tre isocentri sul tronco la coordinata x è uguale a -y nell'Iso successivo. TOT=21
        # CampiY ... XGambe,YGambe, XNull, X8, X1testa, Y1testa, X2testa, Y2testa, Xbraccia, Ybraccia, ... TOT=10
        """
        # Ciclo for inutile, quindi va cambiato
        y_reg = np.zeros(shape=(self.num_patients, 1, 44), dtype=float)
        for i, (iso, jaw_X_pix, jaw_Y_pix) in enumerate(
            zip(isocenters_pix_flat, jaws_X_pix_flat, jaws_Y_pix_flat)
        ):
            unique_iso_idx = [
                0,
                1,
                12,
                13,
                30,
                31,
                33,
            ]
            y_iso_new2 = np.zeros(shape=(6), dtype=float)
            y_iso_new1 = iso[unique_iso_idx]
            for z in range(6):
                y_iso_new2[z] = iso[z * 3 * 2 + 2]
            usless_idx = [11, 15, 19]
            y_jaw_X = np.delete(jaw_X_pix, usless_idx)
            unique_Y_idx = [0, 2, 4, 8, 16, 17, 18, 19, 20, 22]
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
