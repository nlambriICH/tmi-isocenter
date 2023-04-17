import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from src.utils.field_geometry_transf import get_zero_row_idx


class Processing:
    """Processing class to prepare the data and for post-training operations"""

    def __init__(
        self,
        masks: list[np.ndarray],
        isocenters_pix: np.ndarray,
        jaws_X_pix: np.ndarray,
        jaws_Y_pix: np.ndarray,
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

    def resize(self, width_resize=512):
        """Resize all masks to have a height of 512 pixels while preserving the aspect ratio.
        The width of the masks will be adjusted accordingly.

        Args:
            width_resize (int): The desired width of the masks after resizing. Default is 512.

        Raises:
            AssertionError: If not all masks have a height of 512 pixels.

        Returns:
            None.

        This method uses the `iaa.Resize` class from the imgaug library to perform the resizing
        operation. The isocenters and Y-jaw aperture positions are adjusted accordingly.
        Only the Y-jaw aperture needs to be resized as the X aperture is aligned with the
        height dimension of the mask.
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

            # Swap the columns to return at the original dicom's reference sistem
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
        Scale the isocenters_pix, jaws_X_pix, and jaws_Y_pix attributes of the object
        based on the size of the masks in the object.

        Raises:
        - AssertionError: If any 2D mask in the object is not a square matrix.

        Returns:
        - None
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
        Rotates each 2D mask in the instance variable `self.masks` 90 degrees counterclockwise using the imgaug library.
        Also rotates the corresponding isocenters in `self.isocenters_pix` by the same amount.

        Returns:
        --------
        self: object
            The modified object with rotated masks and isocenters.

        Notes:
            For a correct use of this function, we suggest to utilize it only after the _resize() function.
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
        Resize every mask to its original height while preserving the aspect ratio.
        The width of the masks will be adjusted accordingly.

        This method uses the `iaa.Resize` class from the imgaug library to perform the resizing
        operation. The isocenters and Y-jaw aperture positions are adjusted accordingly.
        Only the Y-jaw aperture needs to be resized as the X aperture is aligned with the
        height dimension of the mask.

        Raises:
            AssertionError: If not all masks have a height of 512 pixels.

        Returns:
            None.
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
            # swap the columns to return at the original dicom's reference sistem
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
        Scale the isocenters_pix, jaws_X_pix, and jaws_Y_pix attributes of the object
        based on the size of the masks in the object.

        Raises:
        - AssertionError: If any 2D mask in the object is not a square matrix.

        Returns:
        - None
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
        Rotates each 2D mask in the instance variable `self.masks` 90 degrees clockwise using the imgaug library.
        Also rotates the corresponding isocenters in `self.isocenters_pix` by the same amount.

        Returns:
        --------
        self: object
            The modified object with rotated masks and isocenters.

        Notes:
        For a correct use of this function, we suggest to utilize it only after the resize() function.
        """

        masks_rot = []
        isos_kps_img_rot3D = np.zeros(
            shape=(self.num_patients, self.iso_per_patient, 3)
        )
        rot = iaa.Rot90(k=-1, keep_size=False)

        for i, (mask2d, iso_pix) in enumerate(zip(self.masks, self.isocenters_pix)):
            # swap the columns to return at the original dicom's reference sistem
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
            # swap the columns to return at the original dicom's reference sistem
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

        Applies an augmentation pipeline to the masks, isocenters, and jaw positions stored in the instance attributes.
        The pipeline includes rotation, scaling and resizing operations to obtain 512x512 vertical masks, and corresponding transformations
        of the isocenters and jaw positions. The resulting augmented data is stored back in the instance attributes.


        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            None.

        Notes:
            For a correct use of this function, we suggest to utilize it only to trasform orizontal images in vertical ones.
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
            self._augmenting(
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

        Applies an augmentation pipeline to the masks, isocenters, and jaw positions stored in the instance attributes.
        The pipeline includes rotation and resizing operations to reset the masks, the isocenters and the jaw powitions in the original shape, and corresponding transformations
        of the isocenters and jaw positions. The resulting augmented data is stored back in the instance attributes.

        Raises:
            AssertionError: If any of the masks does not have a height of 512 pixels.

        Returns:
            None.

        Notes:
            For a correct use of this function, we suggest to utilize it only to trasform vertical images in orizontal ones.
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
            # swap the columns to return at the original dicom's reference sistem
            iso_pix[:, [2, 0]] = iso_pix[:, [0, 2]]

            # Keypoint x: column-wise == dicom-z, keypoint y: row-wise == dicom-x
            self._augmenting(
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
            iso_pix[:, [2, 0]] = iso_pix[:, [0, 2]]

        self.masks = masks_aug
        self.isocenters_pix = isos_kps_img_aug3D
        self.jaws_X_pix = self.jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix_aug

        if self.get_masks[0].shape[0] == self.get_masks[0].shape[1]:
            self.scale()

        return self

    def _augmenting(
        self,
        masks_aug,
        isos_kps_img_aug3D,
        jaws_Y_pix_aug,
        i,
        mask2d,
        iso_pix,
        jaw_Y_pix,
        width_resize,
        aug,
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
