import os
import glob
import warnings
import numpy as np
import pandas as pd
import pydicom
from pydicom import Dataset
from rt_utils import RTStructBuilder, RTStruct
from rt_utils.image_helper import (
    get_patient_to_pixel_transformation_matrix,
    apply_transformation_to_3d_points,
    get_slice_directions,
    get_spacing_between_slices,
)
from src.config.constants import (
    PTV_EXCLUDE_SUBSTRINGS,
    MAP_ID_PTV,
    MAP_ID_JUNCTION,
    DICOM_PATH,
)
from src.utils.field_geometry_transf import transform_field_geometry


def filter_ptv_name(name: str) -> bool:
    name_lower = name.lower()
    return ("ptv" in name_lower or "tot" in name_lower) and all(
        substring not in name_lower for substring in PTV_EXCLUDE_SUBSTRINGS
    )


def filter_junction_name(name: str) -> bool:
    name_lower = name.lower()
    return (
        "giun" in name_lower
        or "junc" in name_lower
        or "gy" in name_lower
        and "dose" not in name_lower
        and "boost" not in name_lower
        and "rem" not in name_lower
    )


def get_ptv_mask_3d(
    rtstruct: RTStruct,
    ptv_name: str,
    junction_names: str | list[str] | None = None,
) -> np.ndarray:
    """
    Get a 3D mask of the Planning Target Volume (PTV) and optionally its junctions.

    Parameters:
    - rtstruct: An instance of the RTStruct class containing the RT structure set.
    - ptv_name: The name of the PTV ROI in the RTStruct.
    - junction_names: Optional. A string or list of strings containing the names of the PTV junction ROIs in the RTStruct.
                      If None, an empty mask is created for the junctions.
                      If a string, the whole junction ROI is used.
                      If a list of strings, the union of the junction substructures is used.
                      Default is None.

    Returns:
    - A 3D binary mask of the PTV and its junctions, where 1 indicates the ROI voxels and 0 indicates the non-ROI voxels.

    Raises:
    - Exception: If the type of junction_names is not recognized.
    """

    # Loading the 3D Mask from within the RT Struct
    mask_3d_ptv = rtstruct.get_roi_mask_by_name(ptv_name)

    if junction_names is None:
        mask_3d_junction = np.zeros_like(mask_3d_ptv)  # empty mask
    elif isinstance(junction_names, str):
        mask_3d_junction = rtstruct.get_roi_mask_by_name(
            junction_names
        )  # whole junction
    elif isinstance(junction_names, list):
        mask_3d_junction = np.logical_or.reduce(
            list(map(rtstruct.get_roi_mask_by_name, junction_names))
        )  # union of junction substructures
    else:
        raise Exception(
            f"Type of junction_names not recognized: {type(junction_names)}"
        )

    return mask_3d_ptv | mask_3d_junction


def get_dicom_field_geometry(
    series_data: list[Dataset], ds: Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the raw field geometry of the given RTPLAN DICOM file in the patient coordinate system

    Args:
        series_data (list[Dataset]): list of DICOM datasets corresponding to the CT series that the RTPLAN belongs to
        ds (Dataset): DICOM RTPLAN data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: isocenters, jaw_X, jaw_Y, coll_angle

    Notes:
        For all possible field configurations:
        - isocenters: isocenters' coordinates (mm), shape=(10 + 2, 3), where 10=body fields, 2=arms fields
        - jaw_X: jaw apertures along X (mm), shape=(12, 2)
        - jaw_Y: jaw apertures along Y (mm), shape=(12, 2)
        - coll_angle: collimator angles (deg), shape=(12,)
    """
    body_iso = np.array(
        [
            beam.ControlPointSequence[0].IsocenterPosition
            for beam in ds.BeamSequence
            if np.abs(beam.ControlPointSequence[0].IsocenterPosition[0])
            <= 100  # Exclude isocenters on the arms
        ]
    )

    pat_to_pix_matrix = get_patient_to_pixel_transformation_matrix(series_data)
    body_iso_pixel = np.around(
        apply_transformation_to_3d_points(
            body_iso,
            pat_to_pix_matrix,
        )
    )

    # Sort by z ascending (from pelvis to head)
    sorted_iso_z_idx = body_iso_pixel[:, 2].argsort()
    body_iso = body_iso[sorted_iso_z_idx]

    body_jaw_X = np.array(
        [
            beam_limiting_device_position.LeafJawPositions
            for beam in ds.BeamSequence
            for beam_limiting_device_position in beam.ControlPointSequence[
                0
            ].BeamLimitingDevicePositionSequence
            if beam_limiting_device_position.RTBeamLimitingDeviceType
            in (
                "X",
                "ASYMX",
            )
            and np.abs(beam.ControlPointSequence[0].IsocenterPosition[0])
            <= 100  # Exclude isocenters on the arms
        ]
    )[sorted_iso_z_idx]

    body_jaw_Y = np.array(
        [
            beam_limiting_device_position.LeafJawPositions
            for beam in ds.BeamSequence
            for beam_limiting_device_position in beam.ControlPointSequence[
                0
            ].BeamLimitingDevicePositionSequence
            if beam_limiting_device_position.RTBeamLimitingDeviceType
            in (
                "Y",
                "ASYMY",
            )
            and np.abs(beam.ControlPointSequence[0].IsocenterPosition[0])
            <= 100  # Exclude isocenters on the arms
        ]
    )[sorted_iso_z_idx]

    body_coll_angle = np.array(
        [
            beam.ControlPointSequence[0].BeamLimitingDeviceAngle
            for beam in ds.BeamSequence
            if np.abs(beam.ControlPointSequence[0].IsocenterPosition[0])
            <= 100  # Exclude isocenters on the arms
        ]
    )[sorted_iso_z_idx]

    assert len(body_iso) == len(body_jaw_X) == len(body_jaw_Y) == len(body_coll_angle)

    if len(body_iso) == 8:  # i.e., only 4 unique isocenters (no thorax)
        body_iso = np.insert(body_iso, 4, np.zeros((2, 3)), axis=0)
        body_jaw_X = np.insert(body_jaw_X, 4, np.zeros((2, 2)), axis=0)
        body_jaw_Y = np.insert(body_jaw_Y, 4, np.zeros((2, 2)), axis=0)
        body_coll_angle = np.insert(body_coll_angle, 4, np.zeros(2), axis=0)

    # Retrieve isocenters on the arms
    arms_iso = np.array(
        [
            beam.ControlPointSequence[0].IsocenterPosition
            for beam in ds.BeamSequence
            if np.abs(beam.ControlPointSequence[0].IsocenterPosition[0]) > 100
        ]
    )

    if arms_iso.any():
        arms_iso_pixel = np.around(
            apply_transformation_to_3d_points(arms_iso, pat_to_pix_matrix)
        )

        # Sort by x ascending (left to right)
        sorted_iso_x_idx = arms_iso_pixel[:, 0].argsort()
        arms_iso = arms_iso[sorted_iso_x_idx]
        arms_jaw_X = np.array(
            [
                beam_limiting_device_position.LeafJawPositions
                for beam in ds.BeamSequence
                for beam_limiting_device_position in beam.ControlPointSequence[
                    0
                ].BeamLimitingDevicePositionSequence
                if beam_limiting_device_position.RTBeamLimitingDeviceType
                in (
                    "X",
                    "ASYMX",
                )
                and np.abs(beam.ControlPointSequence[0].IsocenterPosition[0]) > 100
            ]
        )[sorted_iso_x_idx]
        arms_jaw_Y = np.array(
            [
                beam_limiting_device_position.LeafJawPositions
                for beam in ds.BeamSequence
                for beam_limiting_device_position in beam.ControlPointSequence[
                    0
                ].BeamLimitingDevicePositionSequence
                if beam_limiting_device_position.RTBeamLimitingDeviceType
                in (
                    "Y",
                    "ASYMY",
                )
                and np.abs(beam.ControlPointSequence[0].IsocenterPosition[0]) > 100
            ]
        )[sorted_iso_x_idx]
        arms_coll_angle = np.array(
            [
                beam.ControlPointSequence[0].BeamLimitingDeviceAngle
                for beam in ds.BeamSequence
                if np.abs(beam.ControlPointSequence[0].IsocenterPosition[0]) > 100
            ]
        )[sorted_iso_x_idx]

        assert (
            len(arms_iso) == len(arms_jaw_X) == len(arms_jaw_Y) == len(arms_coll_angle)
        )

        isocenters = np.vstack((body_iso, arms_iso))
        jaw_X = np.vstack((body_jaw_X, arms_jaw_X))
        jaw_Y = np.vstack((body_jaw_Y, arms_jaw_Y))
        coll_angle = np.append(body_coll_angle, arms_coll_angle)
    else:
        isocenters = np.vstack((body_iso, np.zeros((2, 3))))
        jaw_X = np.vstack((body_jaw_X, np.zeros((2, 2))))
        jaw_Y = np.vstack((body_jaw_Y, np.zeros((2, 2))))
        coll_angle = np.append(body_coll_angle, np.zeros(2))

    return isocenters, jaw_X, jaw_Y, coll_angle


def get_ptv_image_3d(series_data: list[Dataset], mask_3d: np.ndarray) -> np.ndarray:
    """
    Generate a 3D image from a list of 2D image datasets and apply a mask.

    Args:
        series_data (list[Dataset]): A list of 2D image datasets.
        mask_3d (np.ndarray): A 3D mask array to be applied to the generated image.

    Returns:
        np.ndarray: A 3D image array obtained by stacking the 2D images from the series_data list along the third dimension,
                    multiplied element-wise by the provided mask_3d.
    """
    # Create 3D array
    img_shape = list(series_data[0].pixel_array.shape)
    img_shape.append(len(series_data))
    img_3d = np.zeros(img_shape)

    for i, s in enumerate(series_data):
        img_2d = s.pixel_array
        img_3d[:, :, i] = img_2d

    assert img_3d.shape == mask_3d.shape

    return img_3d * mask_3d


def read_dicoms() -> tuple[list, list, list, list, list, list, list]:
    """
    Read DICOM files and extract relevant raw information for model training.

    Returns:
        Tuple containing the following information:
            - patient_info: List of tuples, where each tuple contains information about a patient, including
                            patient ID, PTV name, junction names, RT plan label, study date, isocenter arm flag,
                            collimator angle flag, and the dimensions of the corresponding mask.
            - masks: List of 2D arrays representing the PTV mask for each patient in the input directory.
            - ptv_imgs: List of 2D arrays representing the PTV density for each patient in the input directory.
            - isocenters_pix: List of 3D arrays representing the isocenter pixel coordinates for each patient.
            - jaws_X_pix: List of 2D arrays representing the X aperture pixel coordinates for each patient.
            - jaws_Y_pix: List of 2D arrays representing the Y aperture pixel coordinates for each patient.
            - angles: List of arrays representing the collimator angles for each patient.

    Raises:
        AssertionError: If the loaded DICOM files do not meet certain requirements, such as all isocenter pixel
                        coordinates being positive or the sign of aperture pixel coordinates not being conserved
                        between DICOM and patient coordinate systems.
    """
    patient_info = []
    masks = []
    ptv_imgs = []
    isocenters_pix = []
    jaws_X_pix = []
    jaws_Y_pix = []
    angles = []

    axis_direction = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    _, patient_dirname, _ = next(os.walk(DICOM_PATH))
    for i, patient_id in enumerate(patient_dirname):
        print(f"Processing patient {i + 1}: {patient_id}")
        dicom_series_path = os.path.join(DICOM_PATH, patient_id)
        rt_struct_path = glob.glob(os.path.join(dicom_series_path, "RTSTRUCT*"))[0]

        # Load existing RT Struct. Requires the series path and existing RT Struct path
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=dicom_series_path,
            rt_struct_path=rt_struct_path,
        )

        # Retrieve the PTV name
        ptv_name = (
            MAP_ID_PTV[patient_id]
            if patient_id in MAP_ID_PTV
            else next(filter(filter_ptv_name, rtstruct.get_roi_names()))
        )

        # Retrieve PTV junction names
        junction_names = (
            MAP_ID_JUNCTION[patient_id]
            if patient_id in MAP_ID_JUNCTION
            else list(filter(filter_junction_name, rtstruct.get_roi_names()))
        )

        mask_3d = get_ptv_mask_3d(
            rtstruct, ptv_name, junction_names
        )  # axis0=y, axis1=x, axis2=z

        rt_plan_path = glob.glob(os.path.join(dicom_series_path, "RTPLAN*"))[0]
        ds = pydicom.read_file(rt_plan_path)

        iso, jaw_X, jaw_Y, coll_angles = get_dicom_field_geometry(
            rtstruct.series_data, ds
        )

        assert np.all(
            [
                x == y
                for (x, y) in zip(
                    get_slice_directions(rtstruct.series_data[0]), axis_direction
                )
            ]
        )

        iso_pixel, jaw_X_pixel, jaw_Y_pixel = transform_field_geometry(
            rtstruct.series_data, iso, jaw_X, jaw_Y
        )

        # iso_pixel coords > 0
        assert np.all(iso_pixel >= 0), "Not all iso pixel were positive"

        # Standard jaw aperture in dicom: (X1, Y1) < 0, (X2, Y2) > 0
        # However, there were cases where the jaw was closed, i.e., X1/X2 < 0 in Eclipse
        # Checking the sign is maintained between dicom and patient's coord systems
        assert np.all(
            np.sign(jaw_X_pixel) == np.sign(jaw_X)
        ), "The sign of X apertures between dicom and patient's coord systems was not conserved"
        assert np.all(
            np.sign(jaw_Y_pixel) == np.sign(jaw_Y)
        ), "The sign of Y apertures between dicom and patient's coord systems was not conserved"

        # Saving the coronal projection (i.e., (x, z) plane) of the PTV mask
        # The 2D mask is the result of the OR operation
        # between the PTV mask of all coronal slices
        # TODO: investigate whether the use of a 3D PTV mask
        # can improve the performance of the model
        mask_2d = mask_3d.any(axis=0)

        # Saving the HU density of PTV in coronal plane (i.e., (x, z) plane)
        # The 2D img is the average of non-zero pixel_array values contained in the PTV along the y axis
        # Taking the mean of empty slices (all elements == 0) gives np.nan and warnings from NumPy
        ptv_img_3d = get_ptv_image_3d(rtstruct.series_data, mask_3d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ptv_img_2d = ptv_img_3d.mean(axis=0, where=ptv_img_3d != 0)
        ptv_img_2d = np.nan_to_num(ptv_img_2d)

        masks.append(mask_2d)
        ptv_imgs.append(ptv_img_2d)
        isocenters_pix.append(iso_pixel)
        jaws_X_pix.append(jaw_X_pixel)
        jaws_Y_pix.append(jaw_Y_pixel)
        angles.append(coll_angles)

        first_slice = rtstruct.series_data[0]
        assert np.all(
            [
                first_slice.PixelSpacing == dcm.PixelSpacing
                for dcm in rtstruct.series_data
            ]
        ), "Not all slices have the same pixel spacing"

        # SliceThickness tag represents the nominal slice thickness in acquisition
        # rt_utils computes the correct slice_spacing using the number of slices and patient orientation
        slice_spacing = get_spacing_between_slices(rtstruct.series_data)

        patient_info.append(
            (
                patient_id,
                ptv_name,
                junction_names,
                ds.RTPlanLabel,
                ds.StudyDate,
                0 if np.all(iso_pixel[-2:] == 0) else 1,  # isocenters on arms
                0
                if np.all(coll_angles[:2] == 90)
                else 1,  # 5/355° coll angle on pelvis
                mask_3d.shape[0],
                mask_3d.shape[1],
                mask_3d.shape[2],
                slice_spacing,
                first_slice.PixelSpacing[0],
            )
        )

    return patient_info, masks, ptv_imgs, isocenters_pix, jaws_X_pix, jaws_Y_pix, angles


if __name__ == "__main__":
    (
        patient_info,
        masks,
        ptv_imgs,
        isocenters_pix,
        jaws_X_pix,
        jaws_Y_pix,
        angles,
    ) = read_dicoms()

    if not os.path.exists(r"data\raw"):
        os.makedirs(r"data\raw")

    pd.DataFrame(
        patient_info,
        columns=(
            "PatientID",
            "PTVID",
            "JunctionIDs",
            "PlanID",
            "PlanDate",
            "IsocenterOnArms",
            "5_355_Deg_CollimatorPelvis",
            "OrigMaskShape_y",
            "OrigMaskShape_x",
            "OrigMaskShape_z",
            "SliceThickness",
            "PixelSpacing",
        ),
    ).to_csv(r"data\patient_info.csv")
    np.savez(
        r"data\raw\masks2D.npz", *masks
    )  # unpack the list to pass the mask2D arrays as positional arguments
    np.savez(
        r"data\raw\ptv_imgs2D.npz", *ptv_imgs
    )  # unpack the list to pass the ptv_imgs2D arrays as positional arguments
    np.save(r"data\raw\isocenters_pix.npy", np.array(isocenters_pix))
    np.save(r"data\raw\jaws_X_pix.npy", np.array(jaws_X_pix))
    np.save(r"data\raw\jaws_Y_pix.npy", np.array(jaws_Y_pix))
    np.save(r"data\raw\angles.npy", np.array(angles))
