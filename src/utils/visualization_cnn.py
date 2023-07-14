import os
from typing import Literal
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data.processing import Processing
from scipy import ndimage  # pyright: ignore[reportGeneralTypeIssues]
from gradient_free_optimizers import GridSearchOptimizer
import yaml
from src.config.constants import MODEL
import lightning.pytorch as pl


class Visualize:
    """Visualization class to visualize and optimize model's output"""

    def __init__(self, log_dic) -> None:
        with np.load(r"data\raw\ptv_masks2D.npz") as npz_masks2d:
            self.ptv_masks = list(npz_masks2d.values())
        with np.load(r"data\raw\ptv_imgs2D.npz") as npz_masks2d:
            self.ptv_hu = list(npz_masks2d.values())
        self.isocenters_pix = np.load(r"data\raw\isocenters_pix.npy")
        self.jaws_X_pix = np.load(r"data\raw\jaws_X_pix.npy")
        self.jaws_Y_pix = np.load(r"data\raw\jaws_Y_pix.npy")
        self.coll_angles = np.load(r"data\raw\angles.npy")
        self.df_patient_info = pd.read_csv(r"data\patient_info.csv")
        self.original_sizes_col_idx = self.df_patient_info.columns.get_loc(
            key="OrigMaskShape_z"
        )
        self.slice_tickness_col_idx = self.df_patient_info.columns.get_loc(
            key="SliceThickness"
        )
        self.model_dir = log_dic

    def build_output(
        self, y_hat: torch.Tensor, patient_idx: int, aspect_ratio: float
    ) -> torch.Tensor:
        output = np.zeros(shape=(84))
        norm = aspect_ratio * self.ptv_hu[patient_idx].shape[1] / 512
        slice_thickness = self.df_patient_info.iloc[
            patient_idx, self.slice_tickness_col_idx
        ]

        # Isocenter indexes
        index_X = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        index_Y = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
        output[index_X] = self.find_x_coord(
            self.ptv_hu[patient_idx]
        )  # x coord repeated 8 times + 2 times for iso thorax
        output[
            index_Y
        ] = 0.5  # y coord repeated 8 times + 2 times for iso thorax, set to 0

        if y_hat.shape[0] == 39:  # whole model
            output[30] = y_hat[0].item()  # x coord right arm
            output[33] = y_hat[1].item()  # x coord left arm

            for z in range(6):  # z coords
                output[z * 3 * 2 + 2] = y_hat[z + 2].item()
                output[z * 3 * 2 + 5] = y_hat[z + 2].item()

            # Begin jaw_X
            for i in range(11):
                output[36 + i] = y_hat[
                    8 + i
                ].item()  # retrieve apertures of first 11 fields

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

        elif y_hat.shape[0] == 30:  # arms model
            original_size = int(
                self.df_patient_info.iloc[
                    patient_idx, self.original_sizes_col_idx
                ]  # pyright: ignore[reportGeneralTypeIssues]
            )
            output[30] = y_hat[0].item()  # z coord right arm
            output[33] = y_hat[
                1
            ].item()  # z coord left arm         #Do I need two coords for the iso z-coord on arms?

            for z in range(2):  # first two z coords
                output[z * 3 * 2 + 2] = y_hat[z + 2].item()
                output[z * 3 * 2 + 5] = y_hat[z + 2].item()
            output[20] = 0  # we skip the third ISO
            output[23] = 0  # we skip the third ISO
            for z in range(3):  # last three z coords we skip the third ISO
                output[(z + 3) * 3 * 2 + 2] = y_hat[z + 4].item()
                output[(z + 3) * 3 * 2 + 5] = y_hat[z + 4].item()

            # Begin jaw_X
            # 4 legs + 3 pelvis
            for i in range(5):
                output[36 + i] = y_hat[
                    7 + i
                ].item()  # retrieve apertures of first 11 fields
            output[42] = y_hat[12].item()
            output[43] = y_hat[13].item()
            # 3 for third iso = null + one symmetric (thus 0 )
            for i in range(3):
                output[44 + i] = 0
            # 3 for chest iso = null + one symmetric (again 0 so)
            output[48] = y_hat[14].item()
            output[50] = y_hat[15].item()  # add in groups of three avoiding repetitions

            for i in range(3):
                output[52 + i] = y_hat[16 + i].item()  # head
                output[56 + i] = y_hat[19 + i].item()  # arms

            # Symmetric apertures
            output[47] = -output[44]
            output[51] = -output[48]
            output[55] = -output[52]

            output[59] = y_hat[22].item()
            # Overlap fields

            output[41] = (y_hat[3].item() - y_hat[4].item() + 0.01) * norm + output[
                50
            ]  # abdomen
            output[49] = (y_hat[4].item() - y_hat[5].item() + 0.03) * norm + output[
                54
            ]  # chest

            # Begin jaw_Y
            for i in range(4):
                if i < 2:
                    # Same apertures opposite signs #LEGS
                    output[60 + 2 * i] = y_hat[i + 23].item()
                    output[61 + 2 * i] = -y_hat[i + 23].item()

                    # 4 fields with equal (and opposite) apertures
                    output[64 + 2 * i] = y_hat[24].item()
                    output[65 + 2 * i] = -y_hat[24].item()
                    output[68 + 2 * i] = 0  # index 35 == thorax iso
                    output[69 + 2 * i] = 0

                    # 2 fields with equal (and opposite) apertures
                    output[72 + 2 * i] = y_hat[24].item()
                    output[73 + 2 * i] = -y_hat[24].item()

                    # Arms apertures with opposite sign
                    output[80 + 2 * i] = -200 / (
                        slice_thickness
                        * original_size  # result of -200 mm/(original shape*slice thickness), fixed aperture normalized
                    )
                    output[81 + 2 * i] = 200 / (slice_thickness * original_size)

                output[76 + i] = y_hat[26 + i].item()  # apertures for the head

        elif y_hat.shape[0] == 25:
            norm = aspect_ratio * self.ptv_hu[patient_idx].shape[1] / 512
            output[30] = 0  # x coord right arm
            output[33] = 0  # x coord left arm

            for z in range(2):  # z coords
                output[z * 3 * 2 + 2] = y_hat[z].item()
                output[z * 3 * 2 + 5] = y_hat[z].item()
                output[(z + 3) * 3 * 2 + 2] = y_hat[z + 2].item()
                output[(z + 3) * 3 * 2 + 5] = y_hat[z + 2].item()
            output[14] = (output[11] + output[20]) / 2
            output[17] = (output[11] + output[20]) / 2
            output[32] = 0  # z coord right arm
            output[35] = 0  # z coord left arm

            # Begin jaw_X
            for i in range(5):
                output[36 + i] = y_hat[4 + i].item()  # 4 legs + down field 4th iso
            for i in range(3):
                output[42 + i] = y_hat[9 + i].item()  # 2 4th iso + down field 3rd iso
                output[52 + i] = y_hat[15 + i].item()  # head fields
                output[56 + i] = 0  # arms fields

            # chest
            output[46] = y_hat[12]  # third iso
            output[48] = y_hat[13]  # chest iso down field
            output[50] = y_hat[14]  # chest iso

            # Overlap fields
            output[41] = (output[8] - output[14] + 0.01) * norm + output[46]  # abdom
            output[45] = (output[14] - output[20] + 0.03) * norm + output[50]  # third
            output[49] = (output[20] - output[26] + 0.02) * norm + output[54]  # chest

            # Symmetric apertures
            output[47] = -output[44]  # third iso
            output[51] = -output[48]  # chest
            output[55] = -output[52]  # head
            output[59] = 0  # arms

            # Begin jaw_Y
            for i in range(4):
                if i < 2:
                    # Same apertures opposite signs LEGS
                    output[60 + 2 * i] = y_hat[i + 18].item()
                    output[61 + 2 * i] = -y_hat[i + 18].item()

                    # 4 fields with equal (and opposite) apertures
                    # Pelvis
                    output[64 + 2 * i] = y_hat[20].item()
                    output[65 + 2 * i] = -y_hat[20].item()
                    # Third iso
                    output[68 + 2 * i] = y_hat[20].item()
                    output[69 + 2 * i] = -y_hat[20].item()
                    # Chest
                    output[72 + 2 * i] = y_hat[20].item()
                    output[73 + 2 * i] = -y_hat[20].item()

                    # Arms apertures with opposite sign
                    output[80 + 2 * i] = 0
                    output[81 + 2 * i] = 0

                output[76 + i] = y_hat[21 + i].item()  # apertures for the head
        path = os.path.join(
            self.model_dir,  # pyright: ignore[reportGeneralTypeIssues, reportOptionalMemberAccess]
            "hparams.yaml",
        )
        with open(path, "r+") as file:
            data = yaml.safe_load(file)
            distance_between_isocenters = (output[2] - output[26]) * (
                slice_thickness * original_size
            )
            dist_pat = "distance between first and last isocenter_" + str(patient_idx)
            data[dist_pat] = float(distance_between_isocenters)
            yaml.dump(data, file)
            data = None

        if output[2] - output[26] > (840 / (slice_thickness * original_size)):
            shift_measure = (output[2] - output[26]) - (
                840 / (slice_thickness * original_size)
            )
            output[[2, 5]] = output[2] - shift_measure
        return torch.from_numpy(output)

    def extract_original_data(
        self,
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

        return isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat

    def add_rectangle_patch(
        self,
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
        if color == "r":
            linestyle = "-"
        else:
            linestyle = "--"
        ax.add_patch(
            mpatches.Rectangle(
                anchor,
                width,
                height,
                angle=angle,
                rotation_point=rotation_point,
                linestyle=linestyle,
                linewidth=0.7,
                edgecolor=color,
                facecolor="none",
            )
        )

    def plot_fields(
        self,
        ax: plt.Axes,
        iso_pixel: np.ndarray,
        jaw_X: np.ndarray,
        jaw_Y: np.ndarray,
        coll_angles: np.ndarray,
        slice_thickness: float,
        pix_spacing: float,
        color: str = "r",
        unit_measure: Literal["pix", "mm"] = "pix",
        single_fig: bool = False,
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
        color_flag = True
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
            if single_fig:
                if color_flag:
                    color = "r"
                else:
                    color = "b"
            self.add_rectangle_patch(
                ax,
                (iso_pixel_col + offset_col, iso_pixel_row - offset_row),
                width,
                height,
                (iso_pixel_col, iso_pixel_row),
                angle,
                color,
            )
            if single_fig:
                color_flag = not color_flag

    def plot_img(
        self,
        img_interim: np.ndarray,
        patient_idx: int,
        output: torch.Tensor,
        path: str,
        coll_angle_hat: torch.Tensor = torch.ones(
            1
        ),  # Default 90, if isn't important visualize the angle.
        mse: torch.Tensor = torch.tensor(0),
        single_fig: bool = False,
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
            mse: torch.Tensor
                MSE loss added in the figure title of separate plots.
            single_fig : bool
                Whether to plot ground thruth and predictions in the same image

        Returns:
            None

        Notes:
        ------
        - The function calls the `extract_data` function to reorganize the CNN output.
        """
        pix_spacing_col_idx = self.df_patient_info.columns.get_loc(key="PixelSpacing")

        pix_spacing = self.df_patient_info.iloc[patient_idx, pix_spacing_col_idx]
        slice_thickness = self.df_patient_info.iloc[
            patient_idx, self.slice_tickness_col_idx
        ]
        aspect_ratio = (
            slice_thickness / pix_spacing
        )  # pyright: ignore[reportGeneralTypeIssues]

        original_size = int(
            self.df_patient_info.iloc[
                patient_idx, self.original_sizes_col_idx
            ]  # pyright: ignore[reportGeneralTypeIssues]
        )

        if output.shape[0] < 84:
            output = self.build_output(output, patient_idx, aspect_ratio)

        isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat = self.extract_original_data(
            output
        )

        isocenters_hat = isocenters_hat[np.newaxis]
        jaws_X_pix_hat = jaws_X_pix_hat[np.newaxis]
        jaws_Y_pix_hat = jaws_Y_pix_hat[np.newaxis]

        angles = 90 * np.ones(12)
        if round(torch.sigmoid(coll_angle_hat).item()) == 1:
            angles[0] = 355
            angles[1] = 5

        if MODEL != "body":
            angles[10] = 355
            angles[11] = 5

        processing_output = Processing(
            [img_interim],  # interim image
            isocenters_hat,
            jaws_X_pix_hat,
            jaws_Y_pix_hat,
            angles,
        )

        # Retrieve information of the original shape
        processing_output.original_sizes = [original_size]
        processing_output.inverse_trasform()

        # Local optimization
        x_right, x_left = self.find_fields_coord(
            patient_idx,
            processing_output.isocenters_pix[0],  # patient's isocenters
        )

        # Shift the arms model isocenters
        if MODEL == "arms" and (
            x_right
            < processing_output.isocenters_pix[0][2, 2]
            < x_right + (x_right - x_left) / 2
            or x_left > processing_output.isocenters_pix[0][2, 2]
        ):
            # Setting the isocenters for arms model at 3/4 space
            processing_output.isocenters_pix[0][2, 2] = (
                x_left + (x_right - x_left) * 3 / 4
            )
            processing_output.isocenters_pix[0][3, 2] = (
                x_left + (x_right - x_left) * 3 / 4
            )
            # Fixing the fields with minimum overlap, after the isocenter shift
            processing_output.jaws_X_pix[0][2, 1] = (
                processing_output.isocenters_pix[0][6, 2]
                - processing_output.isocenters_pix[0][2, 2]
                + 1
            ) * aspect_ratio + processing_output.jaws_X_pix[0][3, 0]

        # Shifting the body model isocenters
        elif MODEL == "body" and (
            x_left - (x_right - x_left) / 2
            < processing_output.isocenters_pix[0][2, 2]
            < x_left + (x_right - x_left) / 2
        ):
            # Shifting the isocenter when it is in the neighborhood above, the jaws are fixed after.
            processing_output.isocenters_pix[0][2, 2] = x_left - 10
            processing_output.isocenters_pix[0][3, 2] = x_left - 10

        # For both the models, if the isocenter is on the backbone.
        if x_left < processing_output.isocenters_pix[0][2, 2] < x_right:
            processing_output.jaws_X_pix[0][2, 0] = (
                (x_left - processing_output.isocenters_pix[0][2, 2] + 1)
                * aspect_ratio
                / 2
            )
            processing_output.jaws_X_pix[0][3, 1] = (
                (x_right - processing_output.isocenters_pix[0][2, 2] - 1)
                * aspect_ratio
                / 2
            )

        self.move_backbone_fields(aspect_ratio, processing_output, x_right, x_left)

        # Set distance between the last two iso only for body model to have symmetric fields
        if MODEL == "body" and processing_output.isocenters_pix[0][2, 2] < x_left:
            translation = 0.5 * (
                (
                    processing_output.isocenters_pix[0][2, 2]
                    + processing_output.jaws_X_pix[0][3, 0] / aspect_ratio
                )
                - (
                    processing_output.isocenters_pix[0][0, 2]
                    + processing_output.jaws_X_pix[0][1, 1] / aspect_ratio
                )
                + (
                    processing_output.isocenters_pix[0][4, 2]
                    + processing_output.jaws_X_pix[0][5, 0] / aspect_ratio
                )
                - (
                    processing_output.isocenters_pix[0][2, 2]
                    + processing_output.jaws_X_pix[0][3, 1] / aspect_ratio
                )
            )

            # Pelvis isocenters
            processing_output.isocenters_pix[0][2, 2] = (
                processing_output.isocenters_pix[0][0, 2]
                + processing_output.jaws_X_pix[0][1, 1] / aspect_ratio
                + translation
                - processing_output.jaws_X_pix[0][3, 0] / aspect_ratio
            )
            processing_output.isocenters_pix[0][3, 2] = (
                processing_output.isocenters_pix[0][0, 2]
                + processing_output.jaws_X_pix[0][1, 1] / aspect_ratio
                + translation
                - processing_output.jaws_X_pix[0][3, 0] / aspect_ratio
            )

            # Abdomen isocenters
            processing_output.isocenters_pix[0][4, 2] = (
                processing_output.isocenters_pix[0][3, 2]
                + processing_output.isocenters_pix[0][6, 2]
            ) / 2
            processing_output.isocenters_pix[0][5, 2] = (
                processing_output.isocenters_pix[0][3, 2]
                + processing_output.isocenters_pix[0][6, 2]
            ) / 2

            self.move_backbone_fields(aspect_ratio, processing_output, x_right, x_left)

        # Move back fields for arms model
        if MODEL == "arms" and processing_output.isocenters_pix[0][2, 2] > x_right:
            processing_output.jaws_X_pix[0][0, 1] = (
                x_right - processing_output.isocenters_pix[0][0, 2] - 1
            ) * aspect_ratio
            processing_output.jaws_X_pix[0][3, 0] = (
                (x_left - processing_output.isocenters_pix[0][2, 2]) + 1
            ) * aspect_ratio

        if single_fig:
            self.single_figure_plot(
                patient_idx,
                path,
                processing_output,
                pix_spacing,  # pyright: ignore[reportGeneralTypeIssues]
                slice_thickness,  # pyright: ignore[reportGeneralTypeIssues]
            )
        else:
            self.separate_plots(
                patient_idx,
                path,
                processing_output,
                pix_spacing,  # pyright: ignore[reportGeneralTypeIssues]
                slice_thickness,  # pyright: ignore[reportGeneralTypeIssues]
                mse=mse,
            )

    def move_backbone_fields(self, aspect_ratio, processing_output, x_right, x_left):
        if processing_output.isocenters_pix[0][2, 2] < x_left:
            processing_output.jaws_X_pix[0][2, 1] = (
                x_right - processing_output.isocenters_pix[0][2, 2] - 1
            ) * aspect_ratio
            processing_output.jaws_X_pix[0][5, 0] = (
                x_left - processing_output.isocenters_pix[0][4, 2] + 1
            ) * aspect_ratio

    def single_figure_plot(
        self,
        patient_idx: int,
        path: str,
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

        plt.imshow(self.ptv_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
        plt.contourf(self.ptv_hu[patient_idx], alpha=0.25)

        plt.scatter(
            processing_output.isocenters_pix[0, :, 2],
            processing_output.isocenters_pix[0, :, 0],
            color="red",
            s=7,
        )
        self.plot_fields(
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
            self.isocenters_pix[patient_idx, :, 2],
            self.isocenters_pix[patient_idx, :, 0],
            color="blue",
            s=7,
        )
        self.plot_fields(
            plt.gca(),
            self.isocenters_pix[patient_idx],
            self.jaws_X_pix[patient_idx],
            self.jaws_Y_pix[patient_idx],
            self.coll_angles[patient_idx],
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

        plt.savefig(
            os.path.join(eval_img_path, f"output_train_{patient_idx}"), dpi=2000
        )
        plt.close()

    def separate_plots(
        self,
        patient_idx: int,
        path: str,
        processing_output: Processing,
        pix_spacing: float,
        slice_thickness: float,
        mse: torch.Tensor = torch.tensor(0),
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
        - mse (torch.Tensor): MSE loss added in the figure title.

        Returns:
        - None: The function saves the plot to disk, then closes it.
        """
        aspect_ratio = slice_thickness / pix_spacing

        # Plot predictions
        plt.imshow(self.ptv_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
        # plt.contourf(processing_raw.masks[patient_idx], alpha=0.25)
        plt.scatter(
            processing_output.isocenters_pix[0, :, 2],
            processing_output.isocenters_pix[0, :, 0],
            color="red",
            s=7,
        )
        self.plot_fields(
            plt.gca(),
            processing_output.isocenters_pix[0],
            processing_output.jaws_X_pix[0],
            processing_output.jaws_Y_pix[0],
            processing_output.coll_angles,
            slice_thickness,
            pix_spacing,
            single_fig=True,
        )
        plt.title(f"MSE loss: {mse}")
        predict_img_path = os.path.join(path, "predict_img")
        if not os.path.exists(predict_img_path):
            os.makedirs(predict_img_path)

        plt.savefig(
            os.path.join(predict_img_path, f"output_test_{patient_idx}"), dpi=2000
        )
        plt.close()

        # Plot ground thruth
        plt.imshow(self.ptv_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
        # plt.contourf(processing_raw.masks[patient_idx], alpha=0.25)
        plt.scatter(
            self.isocenters_pix[patient_idx, :, 2],
            self.isocenters_pix[patient_idx, :, 0],
            color="blue",
            s=7,
        )
        self.plot_fields(
            plt.gca(),
            self.isocenters_pix[patient_idx],
            self.jaws_X_pix[patient_idx],
            self.jaws_Y_pix[patient_idx],
            self.coll_angles[patient_idx],
            slice_thickness,
            pix_spacing,
            "b",
            single_fig=True,
        )

        real_img_path = os.path.join(path, "real_img")
        if not os.path.exists(real_img_path):
            os.makedirs(real_img_path)

        plt.savefig(os.path.join(real_img_path, f"output_test_{patient_idx}"), dpi=2000)
        plt.close()

    def find_x_coord(self, masks_int: np.ndarray) -> float:
        """
            Find the x-coordinate of the center of mass of the given binary masks.

        Args:
            masks_int (np.ndarray): A numpy array representing binary masks.

        Returns:
            float: The normalized x-coordinate of the center of mass.

        """

        x_coord = round(
            ndimage.center_of_mass(masks_int)[
                0
            ]  # pyright: ignore[reportGeneralTypeIssues]
        )

        return x_coord / masks_int.shape[0]  # Normalize the coordinate

    def find_fields_coord(self, patient_index: int, iso: np.ndarray) -> tuple[int, int]:
        iso_on_arms = self.df_patient_info["IsocenterOnArms"].to_numpy().astype(bool)
        ptv_mask = self.ptv_masks[patient_index]

        if iso_on_arms[patient_index]:
            a = (iso[0, 2] + iso[2, 2]) / 2
        else:
            a = (iso[0, 2] + iso[2, 2]) / 2 + 10
        b = (iso[2, 2] + iso[6, 2]) / 2
        search_space = {"x_0": np.arange(a, b, 1, dtype=int)}

        def loss(pos_new):
            x = pos_new["x_0"]
            score = np.sum(ptv_mask[:, x])
            return -score

        opt = GridSearchOptimizer(search_space)
        opt.search(loss, n_iter=search_space["x_0"].shape[0], verbosity=False)

        best_x_pixel = opt.best_value[0]
        x_com = int(self.find_x_coord(self.ptv_hu[patient_index]) * ptv_mask.shape[0])
        y_pixels = np.concatenate(
            (
                np.arange(x_com - 115, x_com - 50),
                np.arange(x_com + 50, x_com + 115),
            )
        )

        min_pos_x_right = best_x_pixel
        min = 512
        for j in y_pixels:
            count = 0
            for i in range(40):
                if not ptv_mask[j, best_x_pixel + i]:  # pixel is background
                    count += 1
                    if (
                        ptv_mask[j, best_x_pixel + i]
                        != ptv_mask[j, best_x_pixel + i + 1]
                    ):
                        break
                if ptv_mask[j, best_x_pixel + i]:  # pixel in mask
                    count = np.inf  # count == np.inf if first pixel is in mask
                    break

            # Assumption: at least one pixel along j is background
            if min > count:
                min = count
                min_pos_x_right = best_x_pixel + count

        assert (
            min_pos_x_right != best_x_pixel
        ), "Optimization has not found the correct position"

        min_pos_x_left = best_x_pixel
        min = 512
        for j in y_pixels:
            count = 0
            for i in range(40):
                if not ptv_mask[j, best_x_pixel - i]:  # pixel is background
                    count += 1
                    if (
                        ptv_mask[j, best_x_pixel - i]
                        != ptv_mask[j, best_x_pixel - i - 1]
                    ):
                        break
                if ptv_mask[j, best_x_pixel - i]:  # pixel in mask
                    count = np.inf  # count == np.inf if first pixel is in mask
                    break

            # Assumption: at least one pixel along j is background
            if min > count:
                min = count
                min_pos_x_left = best_x_pixel - count

        assert (
            min_pos_x_left != best_x_pixel
        ), "Optimization has not found the correct position"

        return (
            min_pos_x_right,
            min_pos_x_left,
        )


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
    test = Visualize()
    reconstructed_output = test.build_output(test_build_output, patient_idx, 4.26)
    assert reconstructed_output.shape[0] == 84

    path = "test"
    # test.plot_img(test.ptv_hu[77],patient_idx, output, path) #NEED TO PASS AN IMAGE INTERIM
