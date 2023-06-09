import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.utils.field_geometry_transf import get_zero_row_idx
import os

"""
Choose between three models:
"body" "arms" "all"
"""
model = "arms"


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
            masks (list[np.ndarray]): List of num_patients arrays of shape (512, height) containing the masks to be resized.
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

        resize = iaa.Resize(size={"height": 512, "width": width_resize})

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
            For a correct use of this function, we suggest to utilize it only with orizontal images.
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
            resize = iaa.Resize(size={"height": 512, "width": width_resize})
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

    def trasform(self):
        """
        Sequence transformation composed of resize, 90 degrees CCW rotation, and scaling of masks and keypoints (isocenters and jaw apertures).

        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Notes:
            This function expects as input masks where height and width correspond to x-axis and z-axis in the patient's coord system ("horizontal" image).
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
                    iaa.Resize(size={"height": 512, "width": width_resize}),
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

    def inverse_trasform(self):
        """
        Sequence transformation composed of scaling, 90 degrees CW rotation, and resize to recover the original values of masks and keypoints (isocenters and jaw apertures).

        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            self: The modified object with rotated masks and isocenters.

        Notes:
            This function expects as input masks where height and width correspond to z-axis and x-axis in the patient's coord system ("vertical" image).
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
                    iaa.Resize(size={"height": 512, "width": width_resize}),
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

        if not os.path.exists(r"data\interim"):
            os.makedirs(r"data\interim")

        np.savez(
            r"data\interim\masks2D.npz", *self.masks
        )  # unpack the list to pass the mask2D arrays as positional arguments
        np.save(r"data\interim\isocenters_pix.npy", self.isocenters_pix)
        np.save(r"data\interim\jaws_X_pix.npy", self.jaws_X_pix)
        np.save(r"data\interim\jaws_Y_pix.npy", self.jaws_Y_pix)
        np.save(r"data\interim\angles.npy", self.coll_angles)


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
        "ptv_imgs": r"data\raw\ptv_imgs2D.npz",
        "ptv_masks": r"data\raw\ptv_masks2D.npz",
        "brain_masks": r"data\raw\brain_masks2D.npz",
        "lungs_masks": r"data\raw\lungs_masks2D.npz",
        "liver_masks": r"data\raw\liver_masks2D.npz",
        "bladder_masks": r"data\raw\bladder_masks2D.npz",
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
    ) in zip(
        loaded_masks["ptv_imgs"],
        loaded_masks["ptv_masks"],
        loaded_masks["brain_masks"],
        loaded_masks["bladder_masks"],
        loaded_masks["lungs_masks"],
        loaded_masks["liver_masks"],
    ):
        channel1 = ptv_img[:, :, np.newaxis]
        channel2 = ptv_mask[:, :, np.newaxis]
        channel3 = 3 * brain_mask[:, :, np.newaxis]
        channel4 = 4 * bladder_mask[:, :, np.newaxis]
        channel5 = 5 * lungs_mask[:, :, np.newaxis]
        channel6 = 6 * liver_mask[:, :, np.newaxis]
        channel_overlap = channel3 + channel4 + channel5 + channel6
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
    patient_info = pd.read_csv(r"data\patient_info.csv")
    iso_on_arms = patient_info["IsocenterOnArms"].to_numpy()
    mask_imgs = load_masks()
    isocenters_pix = np.load(r"data\raw\isocenters_pix.npy")
    jaws_X_pix = np.load(r"data\raw\jaws_X_pix.npy")
    jaws_Y_pix = np.load(r"data\raw\jaws_Y_pix.npy")
    coll_angles = np.load(r"data\raw\angles.npy")

    if model == "arms":
        arms_mask = [mask for mask, bool_val in zip(mask_imgs, iso_on_arms) if bool_val]
        iso_on_arms = iso_on_arms.astype(bool)
        arms_iso = isocenters_pix[iso_on_arms]
        arms_j_X = jaws_X_pix[iso_on_arms]
        arms_j_Y = jaws_Y_pix[iso_on_arms]
        arms_coll_ang = coll_angles[iso_on_arms]
        processing = Processing(
            arms_mask,
            arms_iso,
            arms_j_X,
            arms_j_Y,
            arms_coll_ang,
        )
    elif model == "body":
        iso_on_arms = ~iso_on_arms.astype(bool)
        arms_mask = [mask for mask, bool_val in zip(mask_imgs, iso_on_arms) if bool_val]
        arms_iso = isocenters_pix[iso_on_arms]
        arms_j_X = jaws_X_pix[iso_on_arms]
        arms_j_Y = jaws_Y_pix[iso_on_arms]
        arms_coll_ang = coll_angles[iso_on_arms]
        processing = Processing(
            arms_mask,
            arms_iso,
            arms_j_X,
            arms_j_Y,
            arms_coll_ang,
        )
    else:
        processing = Processing(
            mask_imgs,
            isocenters_pix,
            jaws_X_pix,
            jaws_Y_pix,
            coll_angles,
        )

    processing.trasform().save_data()
