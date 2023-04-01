import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


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
        self.masks = masks
        self.isocenters_pix = isocenters_pix
        self.jaws_X_pix = jaws_X_pix
        self.jaws_Y_pix = jaws_Y_pix
        self.num_patients = len(masks)
        self.iso_per_patient = isocenters_pix.shape[1]

    def resize_scale(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Resize masks and associated data to the same width and height, assuming a height of 512 pixels. Scale the pixel positions to the range [0, 1].

        Args:
            masks (np.ndarray): Array of shape (num_patients, 512, height) containing the masks to be resized.
            isocenters_pix (np.ndarray): Array of shape (num_patients, iso_per_patient, 3) containing the position of the isocenters in pixel coordinates.
            jaws_Y_pix (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the Y-jaws apertures in pixel.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of three numpy arrays:
                masks_resize (np.ndarray): Array of shape (num_patients, 512, 512) containing the resized masks.
                iso_rescaled (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the resized and scaled positions of the isocenters on the image plane.
                jaw_X_rescaled (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the scaled X-jaws apertures in pixel.
                jaw_Y_rescaled (np.ndarray): Array of shape (num_patients, iso_per_patient, 2) containing the resized and scaled Y-jaws apertures in pixel.
        """
        assert np.all(
            [mask.shape[0] == 512 for mask in self.masks]
        ), "Not all masks have height=512"

        width_resize = 512

        masks_aug = np.zeros(shape=(self.num_patients, 512, width_resize))
        isos_kps_img_aug = np.zeros(shape=(self.num_patients, self.iso_per_patient, 2))
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
            masks_aug[i] = mask_aug
            isos_kps_img_aug[i] = iso_kps_img_aug.to_xy_array()

            # Only Y apertures need to be resized (X aperture along x/height)
            jaw_Y_pix_aug = jaw_Y_pix * width_resize / mask2d.shape[1]
            jaws_Y_pix_aug[i] = jaw_Y_pix_aug

        assert (
            masks_aug.shape[1] == masks_aug.shape[2]
        ), "Cannot scale because 2D masks are not square matrices"

        return (
            masks_aug,
            isos_kps_img_aug / width_resize,
            self.jaws_X_pix / width_resize,
            jaws_Y_pix_aug / width_resize,
        )

    def resize_scale_X_y(self):
        """
        Perform resize and scale and return data in an input/target format.
        """
        masks, isos, jaws_X, jaws_Y = self.resize_scale()
        isos_flat = isos.reshape(self.num_patients, -1)
        jaws_X_flat = jaws_X.reshape(self.num_patients, -1)
        jaws_Y_flat = jaws_Y.reshape(self.num_patients, -1)
        return masks, np.concatenate((isos_flat, jaws_X_flat, jaws_Y_flat), axis=1)

    # TODO: implement inverse operations of resize_scale
    # TODO: implement 90° CCW rotation of masks such that the conv starts head first

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
