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
    """
    A utility class for augmenting images created in processing.

    `Augmentation` provides a method to apply various data
    augmentation techniques, including flipping, translation,
    elastic transformation, and cutout.

    Attributes:
        - masks (np.ndarray): The array of images, shape (N, C, H, W).
        - isocenters_pix (np.ndarray): Isocenter positions in pixel coordinates.
        - train_indexes (np.ndarray): Indexes of training images in the dataset.
        - jaws_X_pix (np.ndarray): Jaw X positions in pixel coordinates.
        - jaws_Y_pix (np.ndarray): Jaw Y positions in pixel coordinates.
        - angles (np.ndarray): Angles associated with the images.
        - df_patient_info (pd.DataFrame): DataFrame containing patient information.

    Methods:
        - augment(): Apply flip, translate, elastic, and cutout augmentations to images.
    """

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
        self.train_indexes_aug = self.train_indexes
        self.num_images_to_augment = (
            int(self.num_patients_train * 1.5)
            if MODEL == "body"
            else int(self.num_patients_train * 2.0)
        )

    def augment(
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
        """
        Apply flip, translate, elastic, and cutout augmentations to a subset of images.

        Returns:
            A tuple containing the augmented data:
            - masks: Augmented masks (np.ndarray)
            - isocenters_pix: Augmented isocenter positions (np.ndarray)
            - jaws_X_pix: Augmented jaw X positions (np.ndarray)
            - jaws_Y_pix: Augmented jaw Y positions (np.ndarray)
            - angles: Augmented angles (np.ndarray)
            - df_patient_info_aug: Augmented patient info DataFrame (pd.DataFrame)
            - train_indexes_aug: Augmented train indexes (np.ndarray)

        Note:
            Jaws aperture are not transformed.
        """
        masks_aug = np.zeros(
            shape=(
                self.num_images_to_augment,
                self.masks[0].shape[1],
                self.masks[0].shape[2],
                self.masks[0].shape[0],  # channel last
            )
        )
        isos_kps_img_augm3D = np.zeros(
            shape=(self.num_images_to_augment, self.isocenters_pix[0].shape[0], 3)
        )

        # Sample the indices of the images to be augmented
        image_indices = random.choices(
            list(self.train_indexes), k=self.num_images_to_augment
        )

        # Sequence of transformations from which we sample
        sometimes = lambda aug: iaa.Sometimes(0.7, aug)  # noqa: E731
        seq = iaa.Sequential(
            [
                sometimes(
                    iaa.Fliplr(
                        p=1,
                        seed=42,
                    )
                ),
                sometimes(
                    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
                ),
                sometimes(iaa.ElasticTransformation(alpha=100, sigma=10)),
                sometimes(
                    iaa.Cutout(
                        nb_iterations=(
                            2,
                            5,
                        ),  # pyright: ignore[reportGeneralTypeIssues]
                        size=0.1,
                        squared=False,
                        fill_mode="constant",
                        cval=0,
                    )
                ),
            ]
        )

        # Transform the keypoints back to the original scale
        masks_channel_last = np.transpose(self.masks, (0, 2, 3, 1))
        processing = Processing(
            list(masks_channel_last),
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
        )
        processing.inverse_scale()

        for i, aug_index in enumerate(image_indices):
            mask2d = processing.masks[aug_index]
            iso_pix = processing.isocenters_pix[aug_index]
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

        scaling_class = Processing(
            list(masks_aug),
            isos_kps_img_augm3D,
            self.jaws_X_pix[image_indices],
            self.jaws_Y_pix[image_indices],
            self.angles[image_indices],
        )
        scaling_class.scale()

        # Extend dataset
        self.train_indexes_aug = np.concatenate(
            (self.train_indexes_aug, image_indices), axis=0
        )
        self.masks = np.concatenate(
            (self.masks, np.transpose(masks_aug, (0, 3, 1, 2))), axis=0
        )
        self.isocenters_pix = np.concatenate(
            (self.isocenters_pix, scaling_class.isocenters_pix),
            axis=0,  # pyright: ignore[reportGeneralTypeIssues]
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
        df_patient_info_aug = pd.concat((self.df_patient_info, rows_aug))

        return (
            self.masks,
            self.isocenters_pix,
            self.jaws_X_pix,
            self.jaws_Y_pix,
            self.angles,
            df_patient_info_aug,
            self.train_indexes_aug,
        )
