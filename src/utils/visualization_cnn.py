import os
from typing import Literal
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data.processing import Processing
from src.utils.functions import directory
from scipy import ndimage
from src.config.constants import MODEL
from src.utils.local_optimization import Optimization


class Visualize:
    """Visualization class to visualize model's output"""

    def __init__(self, log_dic) -> None:
        with np.load(directory(r"raw\ptv_masks2D.npz")) as npz_masks2d:
            self.ptv_masks = list(npz_masks2d.values())
        with np.load(directory(r"raw\ptv_imgs2D.npz")) as npz_masks2d:
            self.img_hu = list(npz_masks2d.values())
        self.isocenters_pix = np.load(directory(r"raw\isocenters_pix.npy"))
        self.jaws_X_pix = np.load(directory(r"raw\jaws_X_pix.npy"))
        self.jaws_Y_pix = np.load(directory(r"raw\jaws_Y_pix.npy"))
        self.coll_angles = np.load(directory(r"raw\angles.npy"))
        self.df_patient_info = pd.read_csv(r"data\patient_info.csv")
        self.original_sizes_col_idx = self.df_patient_info.columns.get_loc(
            key="OrigMaskShape_z"
        )
        self.slice_thickness_col_idx = self.df_patient_info.columns.get_loc(
            key="SliceThickness"
        )
        self.model_dir = log_dic

    def build_output(
        self,
        y_hat: torch.Tensor,
        patient_idx: int,
        aspect_ratio: float,
        input_img: np.ndarray,
    ) -> torch.Tensor:
        """
        Build the output tensor based on the predicted values and patient information.

        Parameters:
        - y_hat (torch.Tensor): Predicted tensor from the model.
        - patient_idx (int): Index of the patient in the test set, to be plotted.
        - aspect_ratio (float): Aspect ratio of the images.
        - input_img (np.ndarray): The input image with shape (C, H, W).

        Returns:
        - torch.Tensor: Output tensor representing the treatment plan.

        Note:
        - The function constructs a 1D output tensor with 84 elements.
        """
        output = np.zeros(shape=(84))
        norm = aspect_ratio * self.img_hu[patient_idx].shape[1] / 512
        slice_thickness = float(
            self.df_patient_info.iloc[
                patient_idx, self.slice_thickness_col_idx
            ]  # pyright: ignore[reportGeneralTypeIssues]
        )
        original_size = int(
            self.df_patient_info.iloc[
                patient_idx, self.original_sizes_col_idx
            ]  # pyright: ignore[reportGeneralTypeIssues]
        )

        # Isocenter indexes
        index_X = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        index_Y = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
        x_com = ndimage.center_of_mass(input_img[0])[1]
        # x coord repeated 8 times + 2 times for iso thorax
        output[index_X] = x_com / input_img[0].shape[0]
        # y coord repeated 8 times + 2 times for iso thorax, set to 0
        output[index_Y] = 0.5

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
            norm = aspect_ratio * self.img_hu[patient_idx].shape[1] / 512
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
            output[41] = (output[8] - output[14] + 0.01) * norm + output[46]  # abdomen
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
        input_img: np.ndarray,
        patient_idx: int,
        output: torch.Tensor,
        path: str,
        coll_angle_hat: torch.Tensor = torch.ones(1),
        mse: torch.Tensor = torch.tensor(0),
        single_fig: bool = False,
    ) -> None:
        """
        Generates and saves a plot of two images for a given patient: the original image and a transformed image.

        Parameters:
            input_img : np.ndarray
                The input image with shape (C, H, W).
            patient_idx : int
                The index of the patient in the dataset.
            output : torch.Tensor
                A 1D NumPy array containing the output of the model.
            path : str
                The path where the plot image will be saved.
            coll_angle_hat : torch.Tensor
                The predicted collimator angle for the pelvic fields. Defaults to 1 (90 deg).
            mse : torch.Tensor
                MSE loss added in the figure title of separate plots. Defaults to 0.
            single_fig : bool
                Whether to plot ground truth and predictions in the same image.

        Returns:
            None

        Notes:
        ------
        - The function calls the `extract_data` function to reorganize the CNN output.
        """
        pix_spacing_col_idx = self.df_patient_info.columns.get_loc(key="PixelSpacing")

        pix_spacing = self.df_patient_info.iloc[patient_idx, pix_spacing_col_idx]
        slice_thickness = self.df_patient_info.iloc[
            patient_idx, self.slice_thickness_col_idx
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
            output = self.build_output(output, patient_idx, aspect_ratio, input_img)

        isocenters_hat, jaws_X_pix_hat, jaws_Y_pix_hat = self.extract_original_data(
            output
        )

        isocenters_hat = isocenters_hat[np.newaxis]
        jaws_X_pix_hat = jaws_X_pix_hat[np.newaxis]
        jaws_Y_pix_hat = jaws_Y_pix_hat[np.newaxis]

        angles = 90 * np.ones(12)
        if coll_angle_hat.item() == 1:
            angles[0] = 355
            angles[1] = 5

        if MODEL != "body":
            angles[10] = 355
            angles[11] = 5

        processing_output = Processing(
            [np.transpose(input_img, (1, 2, 0))],
            isocenters_hat,
            jaws_X_pix_hat,
            jaws_Y_pix_hat,
            angles,
        )

        # Retrieve information of the original shape
        processing_output.original_sizes = [original_size]
        processing_output.inverse_transform()

        local_optimization = Optimization(
            patient_idx=patient_idx,
            processing_output=processing_output,
            img=input_img,
            aspect_ratio=aspect_ratio,
        )
        local_optimization.optimize()

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

        plt.imshow(self.img_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
        plt.contourf(self.img_hu[patient_idx], alpha=0.25)

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

        # Plot ground truth
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

        eval_img_path = os.path.join(path, "img", "train")
        if not os.path.exists(eval_img_path):
            os.makedirs(eval_img_path)

        plt.savefig(os.path.join(eval_img_path, f"train_{patient_idx}"))
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
        plt.imshow(self.img_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
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
        plt.title(f"MSE loss: {mse:.6f}")
        predict_img_path = os.path.join(path, "img", "test", "predicted")
        if not os.path.exists(predict_img_path):
            os.makedirs(predict_img_path)

        plt.savefig(os.path.join(predict_img_path, f"test_{patient_idx}"))
        plt.close()

        # Plot ground truth
        plt.imshow(self.img_hu[patient_idx], cmap="gray", aspect=1 / aspect_ratio)
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

        real_img_path = os.path.join(path, "img", "test", "ground_truth")
        if not os.path.exists(real_img_path):
            os.makedirs(real_img_path)

        plt.savefig(os.path.join(real_img_path, f"test_{patient_idx}"))
        plt.close()
