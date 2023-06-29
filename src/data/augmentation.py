"""Dataset utility functions"""
from typing import Literal
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.utils.field_geometry_transf import get_zero_row_idx
from src.data.processing import Processing


class Augmentation:
    """Dataset class to augment data"""

    def __init__(self, masks: list[np.ndarray], train_indexes: np.ndarray) -> None:
        self.masks = masks
        self.isocenters_pix = np.load(
            r"data\interim\isocenters_pix.npy"
        )  # shape=(N, 12, 3)
        self.train_indexes = train_indexes
        self.train_masks = masks[train_indexes]
        self.train_iso = self.isocenters_pix[train_indexes]
        self.num_patients = self.train_masks.shape[0]
        self.jaws_X_pix = np.load(r"data\interim\jaws_X_pix.npy")  # shape=(N, 12, 2)
        self.jaws_Y_pix = np.load(r"data\interim\jaws_Y_pix.npy")  # shape=(N, 12, 2)
        self.angles = np.load(r"data\interim\angles.npy")  # shape=(N, 12)
        self.angle_class = np.where(self.angles[:, 0] == 90, 0.0, 1.0)  # shape=(N,)
        self.df_patient_info = pd.read_csv(r"data\patient_info.csv")

    def flip_translate_augmentation(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
    ]:
        masks_nnc = np.transpose(self.train_masks, (0, 2, 3, 1))
        inverse_scaling_class = Processing(
            list(masks_nnc),
            self.train_iso,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
        )
        inverse_scaling_class.inverse_scale()
        original_dim_iso = inverse_scaling_class.isocenters_pix
        masks_aug = np.zeros(
            shape=(
                self.num_patients,
                self.masks[0].shape[0],
                self.masks[0].shape[1],
                self.masks[0].shape[2],
            )
        )
        isos_kps_img_augm3D = np.zeros(
            shape=(self.num_patients, self.isocenters_pix[0].shape[0], 3)
        )
        seq = iaa.Sequential(
            [
                iaa.Fliplr(
                    p=1,
                    seed=42,
                ),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                iaa.ElasticTransformation(alpha=100, sigma=10),
                iaa.Cutout(
                    nb_iterations=(2, 5),
                    size=0.1,
                    squared=False,
                    fill_mode="constant",
                    cval=0,
                ),
            ]
        )
        for i, (mask2d, iso_pix) in enumerate(zip(self.train_masks, original_dim_iso)):
            iso_kps_img = KeypointsOnImage(
                [Keypoint(x=iso[0], y=iso[2]) for iso in iso_pix],
                shape=mask2d.shape,
            )
            img_augmented, iso_kps_img_augm = seq(
                image=mask2d, keypoints=iso_kps_img
            )  # pyright: ignore[reportOptionalMemberAccess, reportGeneralTypeIssues]
            masks_aug[i] = img_augmented
            isos_kps_temp_augm = (
                iso_kps_img_augm.to_xy_array()  # pyright: ignore[reportOptionalMemberAccess, reportGeneralTypeIssues]
            )
            isos_kps_temp_augm[get_zero_row_idx(iso_pix)] = 0
            # isos_kps_temp_augm[:, [1, 0]] = isos_kps_temp_augm[:, [0, 1]] swap not necessary
            isos_kps_img_augm3D[i] = np.insert(
                isos_kps_temp_augm, 1, iso_pix[:, 1], axis=1
            )
        masks_nnc = np.transpose(masks_aug, (0, 2, 3, 1))
        scaling_class = Processing(
            list(masks_nnc),
            isos_kps_img_augm3D,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
        )
        scaling_class.scale()
        isos_kps_img_augm3D = scaling_class.isocenters_pix
        masks_affine = np.concatenate((self.masks, masks_aug), axis=0)
        isocenters_pix_affine = np.concatenate(
            (self.isocenters_pix, isos_kps_img_augm3D), axis=0
        )
        jaws_X_pix_affine = np.concatenate(
            (self.jaws_X_pix, self.jaws_X_pix[self.train_indexes]), axis=0
        )
        jaws_Y_pix_affine = np.concatenate(
            (self.jaws_Y_pix, self.jaws_Y_pix[self.train_indexes]), axis=0
        )
        angles_affine = np.concatenate(
            (self.angles, self.angles[self.train_indexes]), axis=0
        )
        aug_idx = np.arange(len(self.masks), len(self.masks) + len(self.train_masks))
        train_affine = np.concatenate((self.train_indexes, aug_idx), axis=0)

        # fixing dataset
        rows_aug = self.df_patient_info.iloc[self.train_indexes]
        df_patient_info_aug = self.df_patient_info.append(rows_aug)
        # TO DO
        # Maybe here I can save the new df_patient, thus I can use it in Visualize
        # To print the train images augmented
        return (
            masks_affine,
            isocenters_pix_affine,
            jaws_X_pix_affine,
            jaws_Y_pix_affine,
            angles_affine,
            df_patient_info_aug,
            train_affine,
        )
