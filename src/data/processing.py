import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.utils.field_geometry_transf import get_zero_row_idx
import os


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
            isos_kps_temp = iso_kps_img_aug.to_xy_array()

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
        rot = iaa.Rot90(k=1, keep_size=False)

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
            isos_kps_temp = iso_kps_img_aug.to_xy_array()
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
        rot = iaa.Rot90(k=-1, keep_size=False)

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
    def get_isocenters(self):
        return self.isocenters_pix

    @property
    def get_x_jaws(self) -> np.ndarray:
        return self.jaws_X_pix

    @property
    def get_y_jaws(self) -> np.ndarray:
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
                    iaa.Rot90(k=1, keep_size=False),
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
                    iaa.Rot90(k=-1, keep_size=False),
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
            isos_kps_img_aug3D[i][:, [2, 0]] = isos_kps_img_aug3D[i][:, [0, 2]]

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
        masks_aug.append(mask_aug)
        isos_kps_temp = iso_kps_img_aug.to_xy_array()
        isos_kps_temp[get_zero_row_idx(iso_pix)] = 0

        isos_kps_img_aug3D[i] = np.insert(isos_kps_temp, 1, iso_pix[:, 1], axis=1)

        # Only Y apertures need to be resized (X aperture along x/height)
        jaw_Y_pix_aug = jaw_Y_pix * width_resize / mask2d.shape[1]
        jaws_Y_pix_aug[i] = jaw_Y_pix_aug

    def save_data(self):
        """
        Saves the trasformed masks, isocenters, jaws positions, and angles in the interim folder of the tmi_isocenter/data directory.
        If the interim folder does not exist, it is created.

        Returns:
            self: The object instance.
        """

        if not os.path.exists(r"\tmi_isocenter\data\interim"):
            os.makedirs(r"\tmi_isocenter\data\interim")

        np.savez(
            r"\tmi_isocenter\data\interim\masks2D.npz", *self.masks
        )  # unpack the list to pass the mask2D arrays as positional arguments
        np.save(r"\tmi_isocenter\data\interim\isocenters_pix.npy", self.isocenters_pix)
        np.save(r"\tmi_isocenter\data\interim\jaws_X_pix.npy", self.jaws_X_pix)
        np.save(r"\tmi_isocenter\data\interim\jaws_Y_pix.npy", self.jaws_Y_pix)
        np.save(r"\tmi_isocenter\data\interim\angles.npy", self.coll_angles)
        return self

    # TODO: remove duplicate information of x and z coords.
    # For the moment keep all x- and z-coords of the isocenters:
    # - z coord is the same for isocenter groups
    # - x coord is the same for the isocenters of the body
    def get_relevant_isocenters(self, iso_kps_img: np.ndarray) -> np.ndarray:
        # z coordinates (even indexes) from pelvis to head
        # x coordinate is the same between different isocenters (index 1)
        # except for the isocenters on the arms (indexes 11 and 13)
        relevant_indexes = [1, 11, 13, 0, 2, 4, 6, 8, 10, 12]

        if iso_kps_img.ndim == 2:
            return iso_kps_img.ravel()[relevant_indexes]
        elif iso_kps_img.ndim == 3:
            return iso_kps_img[:, relevant_indexes, :]
        else:
            raise ValueError(
                f"Expected an array of ndim == 2 or 3, but got an array with ndim={iso_kps_img.ndim}"
            )


if __name__ == "__main__":
    with np.load("/tmi_isocenter/data/raw/masks2D.npz") as npz_masks2d:
        masks = list(npz_masks2d.values())
    isocenters_pix = np.load("/tmi_isocenter/data/raw/isocenters_pix.npy")
    jaws_X_pix = np.load("/tmi_isocenter/data/raw/jaws_X_pix.npy")
    jaws_Y_pix = np.load("/tmi_isocenter/data/raw/jaws_Y_pix.npy")
    coll_angles = np.load("/tmi_isocenter/data/raw/angles.npy")
    process = Processing(
        masks,
        isocenters_pix,
        jaws_X_pix,
        jaws_Y_pix,
        coll_angles,
    )
    process.save_data()
