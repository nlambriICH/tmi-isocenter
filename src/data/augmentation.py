"""Dataset utility functions"""
import random
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.utils.field_geometry_transf import get_zero_row_idx
from src.data.processing import Processing
from src.config.constants import MODEL


class Augmentation:
    """Dataset class to augment data"""

    def __init__(
        self,
        masks: np.ndarray,
        train_indexes: np.ndarray,
        isocenters_pix: np.ndarray,
        jaws_X_pix: np.ndarray,
        jaws_Y_pix: np.ndarray,
        angles: np.ndarray,
        df_patient_info: pd.DataFrame,
    ) -> None:
        self.masks = masks
        self.isocenters_pix = isocenters_pix
        self.train_indexes = train_indexes
        self.train_masks = masks[train_indexes]
        self.train_iso = self.isocenters_pix[train_indexes]
        self.num_patients_train = self.train_masks.shape[0]
        self.jaws_X_pix = jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix
        self.angles = angles
        self.angle_class = np.where(self.angles[:, 0] == 90, 0.0, 1.0)
        self.df_patient_info = df_patient_info
        self.train_affine = self.train_indexes
        if MODEL == "body":
            self.num_images_to_augment = int(self.num_patients_train * 1.5)
        else:
            self.num_images_to_augment = int(self.num_patients_train * 2.0)

    def augment_affine(
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
        """Apply flip, translate, elastic augmentations to a subset of images.

        Returns:
            A tuple containing the augmented data:
            - masks: Augmented masks (np.ndarray)
            - isocenters_pix: Augmented isocenter positions (np.ndarray)
            - jaws_X_pix: Augmented jaw X positions (np.ndarray)
            - jaws_Y_pix: Augmented jaw Y positions (np.ndarray)
            - angles: Augmented angles (np.ndarray)
            - df_patient_info_aug: Augmented patient info DataFrame (pd.DataFrame)
            - train_affine: Augmented train affine (np.ndarray)
        """
        masks_nnc = np.transpose(self.masks, (0, 2, 3, 1))
        # Create the class processing to return at the original measures
        inverse_scaling_class = Processing(
            list(masks_nnc),
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
        )
        inverse_scaling_class.inverse_scale()
        # Saving the original isocenter positions
        original_dim_iso = inverse_scaling_class.isocenters_pix
        masks_aug = np.zeros(
            shape=(
                self.num_images_to_augment,
                self.masks[0].shape[0],
                self.masks[0].shape[1],
                self.masks[0].shape[2],
            )
        )
        isos_kps_img_augm3D = np.zeros(
            shape=(self.num_images_to_augment, self.isocenters_pix[0].shape[0], 3)
        )
        # Get the indices of the images to be augmented
        image_indices = random.choices(
            list(self.train_indexes), k=self.num_images_to_augment
        )

        # Array of trasformations from which we sample
        augment = [
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
        for i, aug_index in enumerate(image_indices):
            augment_choice = random.sample(population=augment, k=3)
            seq = iaa.Sequential(augment_choice)
            mask2d = self.masks[aug_index]
            iso_pix = original_dim_iso[aug_index]
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
            isos_kps_img_augm3D[i] = np.insert(
                isos_kps_temp_augm, 1, iso_pix[:, 1], axis=1
            )
        masks_nnc = np.transpose(masks_aug, (0, 2, 3, 1))
        scaling_class = Processing(
            list(masks_nnc),
            isos_kps_img_augm3D,
            self.jaws_X_pix[image_indices],
            self.jaws_Y_pix[image_indices],
            self.angles[image_indices],
        )
        scaling_class.scale()
        isos_kps_img_augm3D = scaling_class.isocenters_pix
        self.train_affine = np.concatenate((self.train_affine, image_indices), axis=0)
        self.masks = np.concatenate((self.masks, masks_aug), axis=0)
        self.isocenters_pix = np.concatenate(
            (self.isocenters_pix, isos_kps_img_augm3D), axis=0
        )
        self.jaws_X_pix = np.concatenate(
            (self.jaws_X_pix, self.jaws_X_pix[image_indices]), axis=0
        )
        self.jaws_Y_pix = np.concatenate(
            (self.jaws_Y_pix, self.jaws_Y_pix[image_indices]), axis=0
        )
        self.angles = np.concatenate((self.angles, self.angles[image_indices]), axis=0)

        # fixing dataset
        rows_aug = self.df_patient_info.iloc[image_indices]
        df_patient_info_aug = self.df_patient_info.append(rows_aug)

        return (
            self.masks,
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
            df_patient_info_aug,
            self.train_affine,
        )
