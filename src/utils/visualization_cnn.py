import os
from typing import Literal
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data.processing import Processing


with np.load(r"data\raw\ptv_imgs2D.npz") as npz_masks2d:
    masks = list(npz_masks2d.values())
with np.load(r"data\interim\masks2D.npz") as npz_masks2d:
    masks_int = list(npz_masks2d.values())
isocenters_pix = np.load(r"data\raw\isocenters_pix.npy")
jaws_X_pix = np.load(r"data\raw\jaws_X_pix.npy")
jaws_Y_pix = np.load(r"data\raw\jaws_Y_pix.npy")
coll_angles = np.load(r"data\raw\angles.npy")
df_patient_info = pd.read_csv(r"data\patient_info.csv")

pix_spacing_col_idx = df_patient_info.columns.get_loc(key="PixelSpacing")
slice_tickness_col_idx = df_patient_info.columns.get_loc(key="SliceThickness")


def build_output(y_hat: torch.Tensor, patient_idx: int) -> torch.Tensor:
    output = np.zeros(shape=(84))

    # Isocenter indexes
    index_X = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    index_Y = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    output[index_X] = find_x_coord(
        masks[patient_idx]
    )  # x coord repeated 8 times + 2 times for iso thorax
    output[
        index_Y
    ] = 100  # y coord repeated 8 times + 2 times for iso thorax, setted to 0
    output[30] = y_hat[0].item()  # x coord right arm
    output[[31, 34]] = 100  # y coord for arms repeated twice, also constrained to 0
    output[33] = y_hat[1].item()  # x coord left arm

    for z in range(6):  # z coords
        output[z * 3 * 2 + 2] = y_hat[z + 2].item()
        output[z * 3 * 2 + 5] = y_hat[z + 2].item()

    # Begin jaw_X
    for i in range(11):
        output[36 + i] = y_hat[8 + i].item()  # retrieve apertures of first 11 fields

    for i in range(3):
        output[48 + i] = y_hat[
            19 + i
        ].item()  # add in groups of three avoiding repetitions
        output[52 + i] = y_hat[22 + i].item()
        output[56 + i] = y_hat[25 + i].item()

    # Symmetric apertures
    output[47] = -output[44]
    output[51] = -output[48]
    output[55] = -output[52]

    output[59] = y_hat[28].item()

    # Begin jaw_Y
    for i in range(4):
        if i < 2:
            # Same apertures opposite signs
            output[60 + 2 * i] = y_hat[i + 29].item()
            output[61 + 2 * i] = -y_hat[i + 29].item()

            # 4 fields with equal (and opposite) apertures
            output[64 + 2 * i] = y_hat[31].item()
            output[65 + 2 * i] = -y_hat[31].item()
            output[68 + 2 * i] = y_hat[32].item()  # index 35 == thorax iso
            output[69 + 2 * i] = -y_hat[32].item()

            # 2 fields with equal (and opposite) apertures
            output[72 + 2 * i] = y_hat[31].item()
            output[73 + 2 * i] = -y_hat[31].item()

            # Arms apertures with opposite sign
            output[80 + 2 * i] = y_hat[37 + i].item()
            output[81 + 2 * i] = -y_hat[37 + i].item()

        output[76 + i] = y_hat[33 + i].item()  # apertures for the head

    return torch.Tensor(output)


def extract_original_data(
    output: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorganize the data from a given output torch.Tensor, returns them in three separate 2D arrays.

    Parameters:
        output : torch.Tensor
            A 1D NumPy array containing the data to be extracted. The expected length of this array is 84.

    Returns:
        tuple of 3 numpy.ndarray
            A tuple containing the following 2D NumPy arrays:
            - isocenters_hat: a 12x3 array containing the extracted isocenter positions in x, y, and z dimensions.
            - jaws_X_pix_hat: a 12x2 array containing the extracted X-jaw positions in left and right directions.
            - jaws_Y_pix_hat: a 12x2 array containing the extracted Y-jaw positions in inferior and superior directions.

    Notes:
    ------
    - The function assumes that the `output` array is of length 84, which is the expected size of the relevant data.
    """
    isocenters_hat = np.zeros((12, 3))
    jaws_X_pix_hat = np.zeros((12, 2))
    jaws_Y_pix_hat = np.zeros((12, 2))
    for i in range(12):
        for j in range(3):
            isocenters_hat[i, j] = output[3 * i + j]
            if j < 2:
                jaws_X_pix_hat[i, j] = output[36 + 2 * i + j]
                jaws_Y_pix_hat[i, j] = output[60 + 2 * i + j]

    return (isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat)


def add_rectangle_patch(
    ax: plt.Axes,
    anchor: tuple[float, float],
    width: float,
    height: float,
    rotation_point: tuple[float, float],
    angle: float,
    color: str = "r",
) -> None:
    """
    Adds a rectangle patch to the given Matplotlib Axes object.

    Parameters:
        ax (plt.Axes): The Matplotlib Axes object to add the rectangle patch to.
        anchor (tuple[float, float]): The (x, y) coordinates of the bottom-left corner of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        rotation_point (tuple[float, float]): The (x, y) coordinates of the point around which to rotate the rectangle.
        angle (float): The angle (in degrees) by which to rotate the rectangle around the rotation point.
        color (str, optional): The color of the rectangle's edge. Defaults to "r" (red).

    Returns:
        None
    """
    ax.add_patch(
        mpatches.Rectangle(
            anchor,
            width,
            height,
            angle=angle,
            rotation_point=rotation_point,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
    )


def plot_fields(
    ax: plt.Axes,
    iso_pixel: np.ndarray,
    jaw_X: np.ndarray,
    jaw_Y: np.ndarray,
    coll_angles: np.ndarray,
    slice_thickness: float,
    pix_spacing: float,
    color: str = "r",
    unit_measure: Literal["pix", "mm"] = "pix",
) -> None:
    """
    Plots rectangular fields on a given Matplotlib Axes object, based on information about each field's isocenter,
    jaw position, and collimator angle.

    Parameters:
        ax (plt.Axes): The Matplotlib Axes object to plot the fields on.
        iso_pixel (np.ndarray): A 2D NumPy array containing the (row, col, depth) coordinates of each field's isocenter.
        jaw_X (np.ndarray): A 2D NumPy array containing the X positions of each field's jaw edges.
        jaw_Y (np.ndarray): A 2D NumPy array containing the Y positions of each field's jaw edges.
        coll_angles (np.ndarray): A 1D NumPy array containing the collimator angles of each field.
        slice_thickness (float): The thickness of the slice being plotted (in units of the `unit_measure` parameter).
        pix_spacing (float): The pixel spacing of the slice being plotted (in units of the `unit_measure` parameter).
        color (str, optional): The color of the rectangle's edge. Defaults to "r" (red).
        unit_measure (str, optional): The units of the `slice_thickness` and `pix_spacing` parameters.
            Must be "pix" or "mm". Defaults to "pix".

    Returns:
        None

    Raises:
        ValueError: If `unit_measure` is not "pix" or "mm".
    """
    aspect_ratio = slice_thickness / pix_spacing
    for i, (iso, X, Y, angle) in enumerate(
        zip(
            iso_pixel,
            jaw_X,
            jaw_Y,
            coll_angles,
        )
    ):
        if all(iso == 0):
            continue  # isocenter not present, skip field
        iso_pixel_col, iso_pixel_row = iso[2], iso[0]
        if unit_measure == "pix":
            offset_col = Y[0]
            offset_row = X[1]
            width = Y[1] - Y[0]
            height = X[1] - X[0]
        elif unit_measure == "mm":
            offset_col = Y[0] / slice_thickness
            offset_row = X[1] / pix_spacing
            width = (Y[1] - Y[0]) / slice_thickness
            height = (X[1] - X[0]) / pix_spacing
        else:
            raise ValueError(
                f'unit_measure must be "pix" or "mm" but was {unit_measure}'
            )

        if angle != 90:
            print(
                f"Collimator angle for field {i + 1} was {angle}°. Plotting with angle=0° for visualization"
            )
            angle = 0
        elif angle == 90:
            offset_col *= aspect_ratio
            offset_row /= aspect_ratio
            width *= aspect_ratio
            height /= aspect_ratio
        add_rectangle_patch(
            ax,
            (iso_pixel_col + offset_col, iso_pixel_row - offset_row),
            width,
            height,
            (iso_pixel_col, iso_pixel_row),
            angle,
            color,
        )


def plot_img(
    patient_idx: int, output: torch.Tensor, path: str, single_fig: bool = False
) -> None:
    """
    Generates and saves a plot of two images for a given patient: the original image and a transformed image.

    Parameters:
        patient_idx : int
            The index of the patient to plot.
        output : torch.Tensor
            A 1D NumPy array containing the output of a model for the specified patient.
        path : str
            The path where the plot image will be saved.
        single_fig : bool
            Whether to plot ground thruth and predictions in the same image

    Returns:
        None

    Notes:
    ------
    - The function calls the `extract_data` function to reorganize the CNN output.
    """
    if output.shape[0] < 84:
        output = build_output(output, patient_idx)

    pix_spacing_col_idx = df_patient_info.columns.get_loc(key="PixelSpacing")
    slice_tickness_col_idx = df_patient_info.columns.get_loc(key="SliceThickness")
    pix_spacing = df_patient_info.iloc[patient_idx, pix_spacing_col_idx]
    slice_thickness = df_patient_info.iloc[patient_idx, slice_tickness_col_idx]

    isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat = extract_original_data(output)

    processing_raw = Processing(
        masks,
        isocenters_pix,
        jaws_X_pix,
        jaws_Y_pix,
        coll_angles,
    )
    processing_raw.resize()
    processing_raw.rotate_90()
    isocenters_hat = isocenters_hat[np.newaxis]
    jaws_X_pix_hat = jaws_X_pix_hat[np.newaxis]
    jaws_Y_pix_hat = jaws_Y_pix_hat[np.newaxis]

    processing_output = Processing(
        [processing_raw.masks[patient_idx]],  # resized mask
        isocenters_hat,
        jaws_X_pix_hat,
        jaws_Y_pix_hat,
        coll_angles[patient_idx],
    )

    # Retrieve information of the original shape
    processing_output.original_sizes = [processing_raw.original_sizes[patient_idx]]

    processing_output.inverse_scale()
    processing_output.inverse_rotate_90()
    processing_output.inverse_resize()
    processing_raw.inverse_rotate_90()
    processing_raw.inverse_resize()

    if single_fig:
        single_figure_plot(
            patient_idx,
            path,
            processing_raw,
            processing_output,
            pix_spacing,  # pyright: ignore[reportGeneralTypeIssues]
            slice_thickness,  # pyright: ignore[reportGeneralTypeIssues]
        )
    else:
        separate_plots(
            patient_idx,
            path,
            processing_raw,
            processing_output,
            pix_spacing,  # pyright: ignore[reportGeneralTypeIssues]
            slice_thickness,  # pyright: ignore[reportGeneralTypeIssues]
        )


def single_figure_plot(
    patient_idx: int,
    path: str,
    processing_raw: Processing,
    processing_output: Processing,
    pix_spacing: float,
    slice_thickness: float,
) -> None:
    """
    Plot the predicted and true isocenters, jaws, and mask of a single patient, overlaying the predicted
    isocenters and jaws on the true data.

    Args:
    - patient_idx (int): Index of the patient to plot.
    - path (str): Path where to save the plot.
    - processing (Processing object): Processing object containing the true mask, isocenters, jaws, and
    collimator angles.
    - test (Processing object): Processing object containing the predicted isocenters, jaws, and collimator
    angles.
    - pix_spacing (float): Pixel spacing of the CT images.
    - slice_thickness (float): Slice thickness of the CT images.

    Returns:
    - None: The function saves the plot to disk, then closes it.
    """
    aspect_ratio = slice_thickness / pix_spacing

    plt.imshow(processing_raw.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    plt.contourf(processing_raw.masks[patient_idx], alpha=0.25)

    plt.scatter(
        processing_output.isocenters_pix[0, :, 2],
        processing_output.isocenters_pix[0, :, 0],
        color="red",
        s=10,
    )
    plot_fields(
        plt.gca(),
        processing_output.isocenters_pix[0],
        processing_output.jaws_X_pix[0],
        processing_output.jaws_Y_pix[0],
        processing_output.coll_angles,
        slice_thickness,
        pix_spacing,
    )

    # Plot ground thruth
    plt.scatter(
        processing_raw.isocenters_pix[patient_idx, :, 2],
        processing_raw.isocenters_pix[patient_idx, :, 0],
        color="blue",
        s=10,
    )
    plot_fields(
        plt.gca(),
        processing_raw.isocenters_pix[patient_idx],
        processing_raw.jaws_X_pix[patient_idx],
        processing_raw.jaws_Y_pix[patient_idx],
        coll_angles[patient_idx],
        slice_thickness,
        pix_spacing,
        "b",
    )

    red_patch = mpatches.Patch(color="red", label="Pred")
    blue_patch = mpatches.Patch(color="blue", label="Real")
    plt.legend(handles=[red_patch, blue_patch], loc=0, frameon=True)

    eval_img_path = os.path.join(path, "eval_img")
    if not os.path.exists(eval_img_path):
        os.makedirs(eval_img_path)

    plt.savefig(os.path.join(eval_img_path, f"output_test_{patient_idx}"), dpi=2000)
    plt.close()


def separate_plots(
    patient_idx: int,
    path: str,
    processing_raw: Processing,
    processing_output: Processing,
    pix_spacing: float,
    slice_thickness: float,
) -> None:
    """
    Plot the predicted and true isocenters, jaws, and mask of a single patient, in two different files png,
    one for the real and another for the predicted one.

    Args:
    - patient_idx (int): Index of the patient to plot.
    - path (str): Path where to save the plot.
    - processing (Processing object): Processing object containing the true mask, isocenters, jaws, and
    collimator angles.
    - test (Processing object): Processing object containing the predicted isocenters, jaws, and collimator
    angles.
    - pix_spacing (float): Pixel spacing of the CT images.
    - slice_thickness (float): Slice thickness of the CT images.

    Returns:
    - None: The function saves the plot to disk, then closes it.
    """
    aspect_ratio = slice_thickness / pix_spacing

    # Plot predictions
    plt.imshow(processing_raw.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    # plt.contourf(processing_raw.masks[patient_idx], alpha=0.25)
    plt.scatter(
        processing_output.isocenters_pix[0, :, 2],
        processing_output.isocenters_pix[0, :, 0],
        color="red",
        s=10,
    )
    plot_fields(
        plt.gca(),
        processing_output.isocenters_pix[0],
        processing_output.jaws_X_pix[0],
        processing_output.jaws_Y_pix[0],
        processing_output.coll_angles,
        slice_thickness,
        pix_spacing,
    )
    predict_img_path = os.path.join(path, "predict_img")
    if not os.path.exists(predict_img_path):
        os.makedirs(predict_img_path)

    plt.savefig(os.path.join(predict_img_path, f"output_test_{patient_idx}"), dpi=2000)
    plt.close()

    # Plot ground thruth
    plt.imshow(processing_raw.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    # plt.contourf(processing_raw.masks[patient_idx], alpha=0.25)
    plt.scatter(
        processing_raw.isocenters_pix[patient_idx, :, 2],
        processing_raw.isocenters_pix[patient_idx, :, 0],
        color="blue",
        s=10,
    )
    plot_fields(
        plt.gca(),
        processing_raw.isocenters_pix[patient_idx],
        processing_raw.jaws_X_pix[patient_idx],
        processing_raw.jaws_Y_pix[patient_idx],
        coll_angles[patient_idx],
        slice_thickness,
        pix_spacing,
        "b",
    )

    real_img_path = os.path.join(path, "real_img")
    if not os.path.exists(real_img_path):
        os.makedirs(real_img_path)

    plt.savefig(os.path.join(real_img_path, f"output_test_{patient_idx}"), dpi=2000)
    plt.close()


def find_x_coord(masks_int: np.ndarray) -> float:
    """
    Calculates the normalized x-coordinate based on non-zero values in a 2D array.

    Args:
        masks_int (ndarray): A 2D NumPy array representing the input masks.

    Returns:
        float or None: The normalized x-coordinate if non-zero values are found in the array, or None otherwise.

    The function iterates over the rows of the input array and identifies the first and last row
    that contain more than 2 non-zero values. It then calculates the midpoint between these rows
    and normalizes it based on the total number of rows in the array.

    If no non-zero values are found, the function returns float("inf").

    """
    first_non_zero_value = float("inf")
    last_non_zero_value = float("-inf")

    for i, row in enumerate(masks_int):
        if row.sum() > 2:
            first_non_zero_value = min(first_non_zero_value, i)
            last_non_zero_value = max(last_non_zero_value, i)

    if first_non_zero_value == float("inf"):
        return float("inf")  # Return float('inf') if all row.sum() <= 2

    x_coord = (first_non_zero_value + last_non_zero_value) / 2

    return x_coord / masks_int.shape[0]  # Normalize the coordinate


if __name__ == "__main__":
    # Output tensor of old CNN model with 30 epochs training
    patient_idx = 77
    output = torch.tensor(  # old output shape (84 numbers)
        [
            5.0588e-01,
            4.9987e-01,
            2.7237e-01,
            4.9922e-01,
            4.9281e-01,
            2.7689e-01,
            5.0469e-01,
            4.9276e-01,
            4.0748e-01,
            4.9819e-01,
            4.9476e-01,
            4.0605e-01,
            4.9738e-01,
            4.9431e-01,
            5.4956e-01,
            4.9909e-01,
            4.9260e-01,
            5.4846e-01,
            5.0365e-01,
            4.9595e-01,
            7.0933e-01,
            5.0560e-01,
            4.9706e-01,
            7.1214e-01,
            5.0432e-01,
            4.9824e-01,
            8.7026e-01,
            5.0043e-01,
            4.9352e-01,
            8.6910e-01,
            4.0263e-03,
            8.6793e-04,
            2.0613e-03,
            -2.2139e-03,
            -1.4555e-03,
            -1.0349e-04,
            -1.3892e-02,
            1.8421e-01,
            -2.8368e-01,
            1.9689e-02,
            -1.4855e-02,
            1.8129e-01,
            -1.8445e-01,
            1.6081e-02,
            -3.4334e-02,
            1.7798e-01,
            -1.5245e-01,
            4.7227e-02,
            -1.5162e-02,
            1.7590e-01,
            -2.0080e-01,
            1.7991e-02,
            -1.3397e-02,
            1.7829e-01,
            -1.8128e-01,
            1.7314e-02,
            -5.4344e-04,
            5.9771e-04,
            4.1955e-03,
            1.8666e-04,
            -1.5276e-01,
            1.5309e-01,
            -1.4945e-01,
            1.5163e-01,
            -1.5083e-01,
            1.5180e-01,
            -1.5181e-01,
            1.5361e-01,
            -1.5326e-01,
            1.5264e-01,
            -1.5179e-01,
            1.5012e-01,
            -1.5294e-01,
            1.5167e-01,
            -1.4822e-01,
            1.5724e-01,
            -8.4362e-02,
            9.3764e-02,
            -8.8720e-02,
            8.2712e-02,
            1.9370e-04,
            -2.8278e-03,
            -2.8790e-03,
            -1.2458e-03,
        ]
    )

    test_build_output = torch.arange(1, 43)  # range [1, 42] step=1, new output shape
    reconstructed_output = build_output(test_build_output, patient_idx)
    assert reconstructed_output.shape[0] == 84

    path = "test"
    plot_img(patient_idx, output, path)
