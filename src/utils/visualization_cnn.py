import os
import torch
from src.data.processing import Processing
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# 30 Epochs training

# Non ha senso importare i dati ogni volta, ma dove lo posso fare?
with np.load(r"data\raw\masks2D.npz") as npz_masks2d:
    masks = list(npz_masks2d.values())
isocenters_pix = np.load(r"data\raw\isocenters_pix.npy")
jaws_X_pix = np.load(r"data\raw\jaws_X_pix.npy")
jaws_Y_pix = np.load(r"data\raw\jaws_Y_pix.npy")
coll_angles = np.load(r"data\raw\angles.npy")
df_patient_info = pd.read_csv(r"data\patient_info.csv").sort_values(by="PlanDate")


def extract_data(output):
    isocenters_hat = np.zeros((12, 3))
    jaws_X_pix_hat = np.zeros((12, 2))
    jaws_Y_pix_hat = np.zeros((12, 2))
    for i in range(12):
        for j in range(3):
            isocenters_hat[i, j] = output[3 * i + j].item()
        for j in range(2):
            jaws_X_pix_hat[i, j] = output[36 + 2 * i + j].item()
            jaws_Y_pix_hat[i, j] = output[60 + 2 * i + j].item()
    return isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat


def add_rectangle_patch(
    ax: plt.Axes,
    anchor: tuple[float, float],
    width: float,
    height: float,
    rotation_point: tuple[float, float],
    angle: float,
    color: str = "r",
) -> None:
    ax.add_patch(
        Rectangle(
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
    unit_measure: str = "pix",
) -> None:
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


def plot_img(patient_idx, output, path) -> None:
    pix_spacing = df_patient_info.iloc[patient_idx, -1]
    slice_thickness = df_patient_info.iloc[patient_idx, -2]

    isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat = extract_data(output)
    processing = Processing(
        masks,
        isocenters_pix,
        jaws_X_pix,
        jaws_Y_pix,
        coll_angles,
    )

    processing.resize()
    mask_test = [processing.masks[patient_idx]]
    isocenters_hat = isocenters_hat[np.newaxis]
    jaws_X_pix_hat = jaws_X_pix_hat[np.newaxis]
    jaws_Y_pix_hat = jaws_Y_pix_hat[np.newaxis]

    test = Processing(
        mask_test,
        isocenters_hat,
        jaws_X_pix_hat,
        jaws_Y_pix_hat,
        coll_angles[patient_idx],
    )

    # Devo recuperare le informazioni sulla vecchia shape!
    original_shape = [processing.original_sizes[patient_idx]]
    test.original_sizes = original_shape

    test.inverse_scale()
    # test.inverse_rotate_90()  ?
    test.inverse_resize()
    processing.inverse_resize()

    # Start plot
    # overlap_plot(patient_idx, path, processing, test, pix_spacing, slice_thickness,)
    single_plot(
        patient_idx,
        path,
        processing,
        test,
        pix_spacing,
        slice_thickness,
    )


def overlap_plot(
    patient_idx,
    path,
    processing,
    test,
    pix_spacing,
    slice_thickness,
):
    aspect_ratio = slice_thickness / pix_spacing
    plt.imshow(processing.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    plt.contourf(processing.masks[patient_idx], alpha=0.25)
    plt.scatter(
        test.isocenters_pix[0, :, 2], test.isocenters_pix[0, :, 0], color="red", s=10
    )
    plt.scatter(
        processing.isocenters_pix[patient_idx, :, 2],
        processing.isocenters_pix[patient_idx, :, 0],
        color="blue",
        s=10,
    )

    plot_fields(
        plt.gca(),
        test.isocenters_pix[0],
        test.jaws_X_pix[0],
        test.jaws_Y_pix[0],
        test.coll_angles,
        slice_thickness,
        pix_spacing,
    )
    plot_fields(
        plt.gca(),
        processing.isocenters_pix[patient_idx],
        processing.jaws_X_pix[patient_idx],
        processing.jaws_Y_pix[patient_idx],
        coll_angles[patient_idx],
        slice_thickness,
        pix_spacing,
        "b",
    )
    red_patch = mpatches.Patch(color="red", label="Pred")
    blue_patch = mpatches.Patch(color="blue", label="Real")
    plt.legend(handles=[red_patch, blue_patch], loc=0, frameon=True)

    if not os.path.exists(os.path.join(path, "predict_img")):
        os.makedirs(os.path.join(path, "predict_img"))
    plt.savefig(
        os.path.join(path, "predict_img", f"output_test_{patient_idx}"), dpi=2000
    )
    plt.close()


def single_plot(
    patient_idx,
    path,
    processing,
    test,
    pix_spacing,
    slice_thickness,
):
    aspect_ratio = slice_thickness / pix_spacing
    plt.imshow(processing.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    plt.contourf(processing.masks[patient_idx], alpha=0.25)
    plt.scatter(
        test.isocenters_pix[0, :, 2], test.isocenters_pix[0, :, 0], color="red", s=10
    )
    plot_fields(
        plt.gca(),
        test.isocenters_pix[0],
        test.jaws_X_pix[0],
        test.jaws_Y_pix[0],
        test.coll_angles,
        slice_thickness,
        pix_spacing,
    )
    if not os.path.exists(os.path.join(path, "predict_img")):
        os.makedirs(os.path.join(path, "predict_img"))
    plt.savefig(
        os.path.join(path, "predict_img", f"output_test_{patient_idx}"), dpi=2000
    )
    plt.close()

    plt.imshow(processing.masks[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
    plt.contourf(processing.masks[patient_idx], alpha=0.25)
    plt.scatter(
        processing.isocenters_pix[patient_idx, :, 2],
        processing.isocenters_pix[patient_idx, :, 0],
        color="blue",
        s=10,
    )
    plot_fields(
        plt.gca(),
        processing.isocenters_pix[patient_idx],
        processing.jaws_X_pix[patient_idx],
        processing.jaws_Y_pix[patient_idx],
        coll_angles[patient_idx],
        slice_thickness,
        pix_spacing,
        "b",
    )
    if not os.path.exists(os.path.join(path, "real_img")):
        os.makedirs(os.path.join(path, "real_img"))
    plt.savefig(os.path.join(path, "real_img", f"output_test_{patient_idx}"), dpi=2000)
    plt.close()


if __name__ == "__main__":
    patient_idx = 9
    output = [
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
    output = torch.Tensor(output)
    path = "d:/trash/tmi-isocenter/lightning_logs/test"
    plot_img(patient_idx, output, path)
