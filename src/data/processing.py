import os
from os.path import exists, join

import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables import Keypoint, KeypointsOnImage

from src.config.constants import INTERIM_DATA_DIR_PATH, RAW_DATA_DIR_PATH
from src.utils.field_geometry_transf import get_zero_row_idx


class Processing:
    """Processing class to prepare the data and for post-training operations"""

    def __init__(
        self,
        masks: list[np.ndarray],
        isocenters_pix: np.ndarray,
        jaws_X_pix: np.ndarray,
        jaws_Y_pix: np.ndarray,
        coll_angles: np.ndarray,
    ) -> None:
        """
        Args:
            masks (list[np.ndarray]): List of num_patients arrays of shape (512, height, C) containing the image to be resized.
            isocenters_pix (np.ndarray): Array of shape (num_patients, iso_per_patient, 3) containing the isocenters in pixel coordinates.
            jaws_X_pix (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the X-jaws apertures in pixel coordinates.
            jaws_Y_pix (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the Y-jaws apertures in pixel coordinates.
        """
        self.masks = masks.copy()
        self.isocenters_pix = isocenters_pix.copy()
        self.jaws_X_pix = jaws_X_pix.copy()
        self.jaws_Y_pix = jaws_Y_pix.copy()
        self.num_patients = len(masks)
        self.iso_per_patient = isocenters_pix.shape[1]
        self.original_sizes = [mask.shape[1] for mask in masks]
        self.coll_angles = coll_angles.copy()

    def resize(self, width_resize=512):
        """Resize all masks to a width specified by `width_resize`. Isocenter positions and jaw apertures are transformed accordingly.

        Args:
            width_resize (int): The desired width of the masks after resizing. Default is 512.

        Raises:
            AssertionError: If not all masks have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Note:
            Only the Y-jaw aperture needs to be resized as the X aperture is aligned to the height of the mask image.
        """
        assert np.all(
            [mask.shape[0] == 512 for mask in self.masks]
        ), "Not all masks have height=512"

        masks_aug = []
        isos_kps_img_aug3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )

        jaws_Y_pix_aug = np.zeros_like(self.jaws_Y_pix)

        resize = iaa.Resize(
            size={"height": 512, "width": width_resize}, interpolation="nearest"
        )

        for i, (mask2d, iso_pix, jaw_Y_pix) in enumerate(
            zip(self.masks, self.isocenters_pix, self.jaws_Y_pix)
        ):
            # Keypoint x: column-wise == dicom-z, keypoint y: row-wise == dicom-x
            iso_kps_img = KeypointsOnImage(
                [Keypoint(x=iso[2], y=iso[0]) for iso in iso_pix],
                shape=mask2d.shape,
            )

            mask_aug, iso_kps_img_aug = resize.augment(
                image=mask2d, keypoints=iso_kps_img
            )  # pyright: ignore[reportGeneralTypeIssues]
            masks_aug.append(mask_aug)
            isos_kps_temp = (
                iso_kps_img_aug.to_xy_array()  # pyright: ignore[reportGeneralTypeIssues]
            )

            # Swap columns to original dicom coordinate system
            isos_kps_temp[:, [1, 0]] = isos_kps_temp[:, [0, 1]]

            isos_kps_img_aug3D[i] = np.insert(isos_kps_temp, 1, iso_pix[:, 1], axis=1)

            # Only Y apertures need to be resized (X aperture along x/height)
            jaw_Y_pix_aug = jaw_Y_pix * width_resize / mask2d.shape[1]
            jaws_Y_pix_aug[i] = jaw_Y_pix_aug

        self.masks = masks_aug
        self.isocenters_pix = isos_kps_img_aug3D
        self.jaws_X_pix = self.jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix_aug

        return self

    def scale(self):
        """
        Scale isocenters_pix, jaws_X_pix, and jaws_Y_pix based on the width of the masks.

        Raises:
            AssertionError: If any 2D mask in the object is not a square matrix.

        Returns:
            self: The modified object with scaled isocenters and jaw apertures.

        Note:
            Scaling should be performed after Resize of the masks to square matrices.
        """
        assert np.all(
            [mask.shape[0] == mask.shape[1] for mask in self.masks]
        ), "Cannot scale because 2D masks are not square matrices"

        width_resize = self.masks[0].shape[1]

        self.isocenters_pix = self.isocenters_pix / width_resize
        self.jaws_X_pix = self.jaws_X_pix / width_resize
        self.jaws_Y_pix = self.jaws_Y_pix / width_resize

        return self

    def rotate_90(self):
        """
        Rotate all masks and isocenters by 90 degrees counterclockwise.

        Returns:
        self: The modified object with rotated masks and isocenters.

        Notes:
            For a correct use of this function, we suggest to utilize it only with horizontal images.
        """
        masks_rot = []
        isos_kps_img_rot3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )
        rot = iaa.Rot90(k=-1, keep_size=False)

        for i, (mask2d, iso_pix) in enumerate(zip(self.masks, self.isocenters_pix)):
            iso_kps_img = KeypointsOnImage(
                [Keypoint(x=iso[2], y=iso[0]) for iso in iso_pix],
                shape=mask2d.shape,
            )
            img_rotated = rot.augment_image(image=mask2d)
            iso_kps_img_rot = rot.augment_keypoints(iso_kps_img)
            masks_rot.append(img_rotated)
            isos_kps_temp_rot = (
                iso_kps_img_rot.to_xy_array()  # pyright: ignore[reportOptionalMemberAccess, reportGeneralTypeIssues]
            )
            isos_kps_temp_rot[get_zero_row_idx(iso_pix)] = 0
            isos_kps_img_rot3D[i] = np.insert(
                isos_kps_temp_rot, 1, iso_pix[:, 1], axis=1
            )

        self.masks = masks_rot
        self.isocenters_pix = isos_kps_img_rot3D

        return self

    def inverse_resize(self):
        """
        Resize every mask to its original width. Isocenter positions and jaw apertures are transformed accordingly.

        Raises:
            AssertionError: If not all masks have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.
        """
        assert np.all(
            [mask.shape[0] == 512 for mask in self.masks]
        ), "Not all masks have height=512"

        masks_aug = []
        isos_kps_img_aug3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )
        jaws_Y_pix_aug = np.zeros_like(self.jaws_Y_pix)

        for i, (mask2d, iso_pix, jaw_Y_pix, width_resize) in enumerate(
            zip(self.masks, self.isocenters_pix, self.jaws_Y_pix, self.original_sizes)
        ):
            # Keypoint x: column-wise == dicom-z, keypoint y: row-wise == dicom-x
            resize = iaa.Resize(
                size={"height": 512, "width": width_resize}, interpolation="nearest"
            )
            iso_kps_img = KeypointsOnImage(
                [Keypoint(x=iso[2], y=iso[0]) for iso in iso_pix],
                shape=mask2d.shape,
            )

            mask_aug, iso_kps_img_aug = resize.augment(
                image=mask2d, keypoints=iso_kps_img
            )  # pyright: ignore[reportGeneralTypeIssues]
            masks_aug.append(mask_aug)
            isos_kps_temp = (
                iso_kps_img_aug.to_xy_array()  # pyright: ignore[reportGeneralTypeIssues]
            )
            # Swap columns to original dicom coordinate system
            isos_kps_temp[:, [1, 0]] = isos_kps_temp[:, [0, 1]]
            isos_kps_img_aug3D[i] = np.insert(isos_kps_temp, 1, iso_pix[:, 1], axis=1)

            # Only Y apertures need to be resized (X aperture along x/height)
            jaw_Y_pix_aug = jaw_Y_pix * width_resize / mask2d.shape[1]
            jaws_Y_pix_aug[i] = jaw_Y_pix_aug

        self.masks = masks_aug
        self.isocenters_pix = isos_kps_img_aug3D
        self.jaws_X_pix = self.jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix_aug

        return self

    def inverse_scale(self):
        """
        Transform isocenters_pix, jaws_X_pix, and jaws_Y_pix to the original scale based on the width of the masks.

        Raises:
            AssertionError: If any 2D mask in the object is not a square matrix.

        Returns:
            self: The modified object with rotated masks and isocenters.
        """
        assert np.all(
            [mask.shape[0] == mask.shape[1] for mask in self.masks]
        ), "Cannot scale because 2D masks are not square matrices"

        width_resize = self.masks[0].shape[1]

        self.isocenters_pix = self.isocenters_pix * width_resize
        self.jaws_X_pix = self.jaws_X_pix * width_resize
        self.jaws_Y_pix = self.jaws_Y_pix * width_resize

        return self

    def inverse_rotate_90(self):
        """
        Rotate all masks and isocenters by 90 degrees clockwise.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Notes:
            For a correct use of this function, we suggest to utilize it only after the resize() function.
        """
        masks_rot = []
        isos_kps_img_rot3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )
        rot = iaa.Rot90(k=1, keep_size=False)

        for i, (mask2d, iso_pix) in enumerate(zip(self.masks, self.isocenters_pix)):
            # Swap columns to original dicom coordinate system
            iso_pix[:, [2, 0]] = iso_pix[:, [0, 2]]
            iso_kps_img = KeypointsOnImage(
                [Keypoint(x=iso[2], y=iso[0]) for iso in iso_pix],
                shape=mask2d.shape,
            )
            img_rotated = rot.augment_image(image=mask2d)
            iso_kps_img_rot = rot.augment_keypoints(iso_kps_img)
            masks_rot.append(img_rotated)
            isos_kps_temp_rot = (
                iso_kps_img_rot.to_xy_array()  # pyright: ignore[reportOptionalMemberAccess, reportGeneralTypeIssues]
            )
            isos_kps_temp_rot[get_zero_row_idx(iso_pix)] = 0
            # Swap columns to original dicom coordinate system
            isos_kps_temp_rot[:, [1, 0]] = isos_kps_temp_rot[:, [0, 1]]
            isos_kps_img_rot3D[i] = np.insert(
                isos_kps_temp_rot, 1, iso_pix[:, 1], axis=1
            )

        self.masks = masks_rot
        self.isocenters_pix = isos_kps_img_rot3D

        return self

    @property
    def get_masks(self) -> list[np.ndarray]:
        return self.masks

    @property
    def get_isocenters_pix(self) -> np.ndarray:
        return self.isocenters_pix

    @property
    def get_jaws_X_pix(self) -> np.ndarray:
        return self.jaws_X_pix

    @property
    def get_jaws_Y_pix(self) -> np.ndarray:
        return self.jaws_Y_pix

    def transform(self):
        """
        Sequence transformation composed of resize, 90 degrees CCW rotation,
        and scaling of masks and keypoints (isocenters and jaw apertures).

        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Notes:
            This function expects as input masks where height and width correspond to x-axis and z-axis
            in the patient's coord system ("horizontal" image).
        """
        assert np.all(
            [mask.shape[0] == 512 for mask in self.masks]
        ), "Not all masks have height=512"

        masks_aug = []
        isos_kps_img_aug3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )

        jaws_Y_pix_aug = np.zeros_like(self.jaws_Y_pix)
        width_resize = 512
        for i, (mask2d, iso_pix, jaw_Y_pix) in enumerate(
            zip(
                self.masks,
                self.isocenters_pix,
                self.jaws_Y_pix,
            )
        ):
            aug = iaa.Sequential(
                [
                    iaa.Resize(
                        size={"height": 512, "width": width_resize},
                        interpolation="nearest",
                    ),
                    iaa.Rot90(k=-1, keep_size=False),
                ]
            )
            # Keypoint x: column-wise == dicom-z, keypoint y: row-wise == dicom-x
            self._augment(
                masks_aug,
                isos_kps_img_aug3D,
                jaws_Y_pix_aug,
                i,
                mask2d,
                iso_pix,
                jaw_Y_pix,
                width_resize,
                aug,
            )

        self.masks = masks_aug
        self.isocenters_pix = isos_kps_img_aug3D
        self.jaws_X_pix = self.jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix_aug

        self.scale()

        return self

    def inverse_transform(self):
        """
        Sequence transformation composed of scaling, 90 degrees CW rotation, and resize
        to recover the original values of masks and keypoints (isocenters and jaw apertures).

        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Notes:
            This function expects as input masks where height and width correspond to z-axis and x-axis
            in the patient's coord system ("vertical" image).
        """
        assert np.all(
            [mask.shape[0] == 512 for mask in self.masks]
        ), "Not all masks have height=512"

        self.inverse_scale()

        masks_aug = []
        isos_kps_img_aug3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )

        jaws_Y_pix_aug = np.zeros_like(self.jaws_Y_pix)

        for i, (mask2d, iso_pix, jaw_Y_pix, width_resize) in enumerate(
            zip(self.masks, self.isocenters_pix, self.jaws_Y_pix, self.original_sizes)
        ):
            aug = iaa.Sequential(
                [
                    iaa.Rot90(k=1, keep_size=False),
                    iaa.Resize(
                        size={"height": 512, "width": width_resize},
                        interpolation="nearest",
                    ),
                ]
            )
            # Swap columns to original dicom coordinate system
            iso_pix[:, [2, 0]] = iso_pix[:, [0, 2]]

            # Keypoint x: column-wise == dicom-z, keypoint y: row-wise == dicom-x
            self._augment(
                masks_aug,
                isos_kps_img_aug3D,
                jaws_Y_pix_aug,
                i,
                mask2d,
                iso_pix,
                jaw_Y_pix,
                width_resize,
                aug,
            )
            isos_kps_img_aug3D[i, :, [2, 0]] = isos_kps_img_aug3D[i, :, [0, 2]]

        self.masks = masks_aug
        self.isocenters_pix = isos_kps_img_aug3D
        self.jaws_X_pix = self.jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix_aug

        return self

    def _augment(
        self,
        masks_aug: list[np.ndarray],
        isos_kps_img_aug3D: np.ndarray,
        jaws_Y_pix_aug: np.ndarray,
        i: int,
        mask2d: np.ndarray,
        iso_pix: np.ndarray,
        jaw_Y_pix: np.ndarray,
        width_resize: int,
        aug: iaa.Sequential,
    ):
        """
        Apply transformations to a 2D mask and associated keypoints.

        Parameters:
        - masks_aug (list[np.ndarray]): List to store augmented 2D masks.
        - isos_kps_img_aug3D (np.ndarray): Array to store augmented 3D keypoints.
        - jaws_Y_pix_aug (np.ndarray): Array to store augmented Y apertures.
        - i (int): Index indicating the current transformed mask.
        - mask2d (np.ndarray): 2D mask to be augmented.
        - iso_pix (np.ndarray): 2D array containing the original isocenter keypoints.
        - jaw_Y_pix (np.ndarray): 1D array containing original Y apertures.
        - width_resize (int): Width to which Y apertures should be resized.
        - aug (imgaug.augmenters.Sequential): Transformation sequence applied to the mask.

        Note:
        - Transformations are applied to the 2D mask and its associated keypoints (isos_kps_img).
        """

        iso_kps_img = KeypointsOnImage(
            [Keypoint(x=iso[2], y=iso[0]) for iso in iso_pix],
            shape=mask2d.shape,
        )

        mask_aug, iso_kps_img_aug = aug.augment(
            image=mask2d, keypoints=iso_kps_img
        )  # pyright: ignore[reportGeneralTypeIssues]
        masks_aug.append(mask_aug)  # pyright: ignore[reportGeneralTypeIssues]
        isos_kps_temp = (
            iso_kps_img_aug.to_xy_array()  # pyright: ignore[reportGeneralTypeIssues]
        )
        isos_kps_temp[get_zero_row_idx(iso_pix)] = 0

        isos_kps_img_aug3D[i] = np.insert(isos_kps_temp, 1, iso_pix[:, 1], axis=1)

        # Only Y apertures need to be resized (X aperture along x/height)
        jaw_Y_pix_aug = jaw_Y_pix * width_resize / mask2d.shape[1]
        jaws_Y_pix_aug[i] = jaw_Y_pix_aug

    def save_data(self):
        """
        Save processed masks, isocenters, jaws positions, and angles in the interim folder.

        Returns:
            None.
        """

        if not exists(INTERIM_DATA_DIR_PATH):
            os.makedirs(INTERIM_DATA_DIR_PATH)

        np.save(join(INTERIM_DATA_DIR_PATH, "masks2D.npy"), np.array(self.masks))
        np.save(join(INTERIM_DATA_DIR_PATH, "isocenters_pix.npy"), self.isocenters_pix)
        np.save(join(INTERIM_DATA_DIR_PATH, "jaws_X_pix.npy"), self.jaws_X_pix)
        np.save(join(INTERIM_DATA_DIR_PATH, "jaws_Y_pix.npy"), self.jaws_Y_pix)
        np.save(join(INTERIM_DATA_DIR_PATH, "angles.npy"), self.coll_angles)


def load_masks() -> list[np.ndarray]:
    """
    Load and process mask images from the specified paths.

    Returns:
        List[np.ndarray]: A list of processed mask images.

    Raises:
        None.

    The function loads mask images from the following paths:
    - "ptv_imgs": Path to the npz file containing the PTV (Planning Target Volume) images.
    - "ptv_masks": Path to the npz file containing the PTV masks.
    - "brain_masks": Path to the npz file containing the brain masks.
    - "lungs_masks": Path to the npz file containing the lung masks.
    - "liver_masks": Path to the npz file containing the liver masks.
    - "bladder_masks": Path to the npz file containing the bladder masks.

    Each mask image is processed by combining multiple channels:
    - Channel 1: PTV image.
    - Channel 2: PTV mask.
    - Channel 3: Brain mask multiplied by 3.
    - Channel 4: Bladder mask multiplied by 4.
    - Channel 5: Lung mask multiplied by 5.
    - Channel 6: Liver mask multiplied by 6.

    The processed mask images are obtained by concatenating the channels along axis 2.

    The function returns a list of processed mask images.
    """

    raw_data = {
        "ptv_imgs": join(RAW_DATA_DIR_PATH, "ptv_imgs2D.npz"),
        "ptv_masks": join(RAW_DATA_DIR_PATH, "ptv_masks2D.npz"),
        "brain_masks": join(RAW_DATA_DIR_PATH, "brain_masks2D.npz"),
        "lungs_masks": join(RAW_DATA_DIR_PATH, "lungs_masks2D.npz"),
        "liver_masks": join(RAW_DATA_DIR_PATH, "liver_masks2D.npz"),
        "bladder_masks": join(RAW_DATA_DIR_PATH, "bladder_masks2D.npz"),
        "intestine_masks": join(RAW_DATA_DIR_PATH, "intestine_masks2D.npz"),
    }

    loaded_masks = {}
    for key, path in raw_data.items():
        with np.load(path) as npz_data:
            loaded_masks[key] = list(npz_data.values())

    mask_imgs = []
    for (
        ptv_img,
        ptv_mask,
        brain_mask,
        bladder_mask,
        lungs_mask,
        liver_mask,
        intestine_mask,
    ) in zip(
        loaded_masks["ptv_imgs"],
        loaded_masks["ptv_masks"],
        loaded_masks["brain_masks"],
        loaded_masks["bladder_masks"],
        loaded_masks["lungs_masks"],
        loaded_masks["liver_masks"],
        loaded_masks["intestine_masks"],
    ):
        channel1 = ptv_img[:, :, np.newaxis]
        channel2 = 0.3 * ptv_mask[:, :, np.newaxis]
        channel3 = 0.5 * brain_mask[:, :, np.newaxis]
        channel4 = 0.5 * bladder_mask[:, :, np.newaxis]
        channel5 = 0.5 * lungs_mask[:, :, np.newaxis]
        channel6 = 0.5 * liver_mask[:, :, np.newaxis]
        channel7 = intestine_mask[:, :, np.newaxis]
        channel_overlap = channel3 + channel4 + channel5 + channel6 + channel7

        image = np.concatenate(
            (
                channel1,
                channel2,
                channel_overlap,
            ),
            axis=2,
        )
        mask_imgs.append(image)

    return mask_imgs


if __name__ == "__main__":
    mask_imgs = load_masks()
    isocenters_pix = np.load(join(RAW_DATA_DIR_PATH, "isocenters_pix.npy"))
    jaws_X_pix = np.load(join(RAW_DATA_DIR_PATH, "jaws_X_pix.npy"))
    jaws_Y_pix = np.load(join(RAW_DATA_DIR_PATH, "jaws_Y_pix.npy"))
    coll_angles = np.load(join(RAW_DATA_DIR_PATH, "angles.npy"))

    processing = Processing(
        mask_imgs,
        isocenters_pix,
        jaws_X_pix,
        jaws_Y_pix,
        coll_angles,
    )

    processing.transform().save_data()
