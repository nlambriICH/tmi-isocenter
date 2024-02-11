from dataclasses import dataclass, field
from os.path import dirname, join

import numpy as np
import pandas as pd
from gradient_free_optimizers import GridSearchOptimizer, ParallelTemperingOptimizer
from scipy import ndimage

from src.config.constants import COLL_5_355, MODEL, RAW_DATA_DIR_PATH
from src.data.processing import Processing


@dataclass
class OptimizationResult:
    """Result of the local optimization:
    'x' pixels of the maximum extension of the ribs and iliac crests.
    """

    x_pixel_ribs: int = field(default=0)
    x_pixel_iliac: int = field(default=0)


@dataclass
class OptimizationSearchSpace:
    """Search space of the local optimization for iliac crests and ribs
    'x' pixel location.
    """

    x_pixel_left: int = field(default=0)
    x_pixel_right: int = field(default=0)
    y_pixels_right: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=65, dtype=int)
    )
    y_pixels_left: np.ndarray = field(
        default_factory=lambda: np.zeros(shape=65, dtype=int)
    )


class Optimization:
    """Local optimization of the abdomen isocenter and related fields.
    The algorithm performs roughly the following steps:
    1) Approximate the location of the spine between the iliac crests and ribs.
    2) Within an appropriate neighborhood of this location, search the pixels
    corresponding to the maximum extension of the iliac crests and ribs.
    3) Adjust the abdomen isocenter and related fields according to the iliac crests
    and ribs positions.
    """

    def __init__(
        self,
        patient_idx: int,
        processing_output: Processing,
        aspect_ratio: float,
    ) -> None:
        self.df_patient_info = pd.read_csv(
            join(dirname(RAW_DATA_DIR_PATH), "patient_info.csv")
        )
        self.patient_idx = patient_idx
        self.original_sizes_col_idx = self.df_patient_info.columns.get_loc(
            key="OrigMaskShape_z"
        )
        self.slice_thickness_col_idx = self.df_patient_info.columns.get_loc(
            key="SliceThickness"
        )
        self.original_size = int(
            self.df_patient_info.iloc[
                patient_idx, self.original_sizes_col_idx
            ]  # pyright: ignore[reportArgumentType]
        )
        self.slice_thickness = float(
            self.df_patient_info.iloc[
                patient_idx, self.slice_thickness_col_idx
            ]  # pyright: ignore[reportArgumentType]
        )
        self.optimization_result = OptimizationResult()
        self.optimization_search_space = OptimizationSearchSpace()
        self.processing = processing_output
        self.aspect_ratio = aspect_ratio
        self.field_overlap_pixels = 10

    def _adjust_field_geometry_body_5_355(self):
        """Adjust the field geometry predicted by the body model, according to the maximum extension
        of iliac crests and ribs."""
        self.field_overlap_pixels = 40 / self.slice_thickness
        x_iliac = (
            self.optimization_result.x_pixel_iliac
        )  # iliac crests 'x' pixel location
        x_ribs = self.optimization_result.x_pixel_ribs  # ribs pixel 'x' location

        self.processing.isocenters_pix[0][2, 2] = (x_iliac + x_ribs) / 2
        self.processing.isocenters_pix[0][3, 2] = self.processing.isocenters_pix[0][
            2, 2
        ]

        # Fix aperture of pelvis field
        if x_ribs - x_iliac < self.field_overlap_pixels:
            self.processing.jaws_X_pix[0][2, 0] = (
                x_iliac - self.processing.isocenters_pix[0][2, 2]
            ) * self.aspect_ratio
            self.processing.jaws_X_pix[0][3, 1] = (
                x_ribs - self.processing.isocenters_pix[0][2, 2]
            ) * self.aspect_ratio
        else:
            self.processing.jaws_X_pix[0][2, 0] = (
                (-self.field_overlap_pixels) * self.aspect_ratio / 2
            )
            self.processing.jaws_X_pix[0][3, 1] = (
                (self.field_overlap_pixels) * self.aspect_ratio / 2
            )

        # Fix third isocenter
        self.processing.isocenters_pix[0][4, 2] = (
            self.processing.isocenters_pix[0][2, 2]
            + self.processing.isocenters_pix[0][6, 2]
        ) / 2
        self.processing.isocenters_pix[0][5, 2] = self.processing.isocenters_pix[0][
            4, 2
        ]

        # Increase aperture of abdominal field (upper)
        self.processing.jaws_X_pix[0][2, 1] = (
            self.processing.isocenters_pix[0][4, 2]
            - self.processing.isocenters_pix[0][2, 2]
            + self.field_overlap_pixels
        ) * self.aspect_ratio + self.processing.jaws_X_pix[0][5, 0]

        # Fix overlap between abdomen and thorax
        self.processing.jaws_X_pix[0][5, 0] = (
            (
                self.processing.isocenters_pix[0][2, 2]
                - self.processing.isocenters_pix[0][4, 2]
                - self.field_overlap_pixels
            )
            * self.aspect_ratio
            / 2
        )
        self.processing.jaws_X_pix[0][2, 1] = (
            (
                self.processing.isocenters_pix[0][4, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            )
            * self.aspect_ratio
            / 2
        )

        # Fix overlap of thorax field (upper)
        self.processing.jaws_X_pix[0][4, 1] = (
            (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][4, 2]
                + self.field_overlap_pixels
            )
            * self.aspect_ratio
            / 2
        )

        self.processing.jaws_X_pix[0][7, 0] = (
            (
                self.processing.isocenters_pix[0][4, 2]
                - self.processing.isocenters_pix[0][6, 2]
                - self.field_overlap_pixels
            )
            * self.aspect_ratio
            / 2
        )

        # Fix overlap of pelvis and abdomen field
        self.processing.jaws_X_pix[0][3, 0] = (
            self.processing.isocenters_pix[0][0, 2]
            - self.processing.isocenters_pix[0][2, 2]
            + self.processing.jaws_Y_pix[0][0, 1]
            - self.field_overlap_pixels
        ) * self.aspect_ratio

    def _adjust_field_geometry_body(self) -> None:
        """Adjust the field geometry predicted by the body model, according to the maximum extension
        of iliac crests and ribs."""

        x_iliac = (
            self.optimization_result.x_pixel_iliac
        )  # iliac crests 'x' pixel location
        x_ribs = self.optimization_result.x_pixel_ribs  # ribs pixel 'x' location

        if (
            x_iliac - (x_ribs - x_iliac)
            < self.processing.isocenters_pix[0][2, 2]
            < x_iliac + (x_ribs - x_iliac) / 2
        ):
            # Shifting the abdomen and pelvis isocenters toward the feet
            old_pos_abdomen_iso = self.processing.isocenters_pix[0][2, 2].copy()

            self.processing.isocenters_pix[0][2, 2] = (
                x_iliac - (x_ribs - x_iliac) / 2.1 - 10
            )
            self.processing.isocenters_pix[0][3, 2] = (
                x_iliac - (x_ribs - x_iliac) / 2.1 - 10
            )
            self.processing.isocenters_pix[0][0, 2] -= (
                old_pos_abdomen_iso - self.processing.isocenters_pix[0][2, 2]
            ) / 2
            self.processing.isocenters_pix[0][1, 2] -= (
                old_pos_abdomen_iso - self.processing.isocenters_pix[0][2, 2]
            ) / 2

        # Isocenter on the spine
        if x_iliac < self.processing.isocenters_pix[0][2, 2] < x_ribs:
            self.processing.jaws_X_pix[0][2, 0] = (
                (x_iliac - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )
            self.processing.jaws_X_pix[0][3, 1] = (
                (x_ribs - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )

            # Increase aperture of abdominal field (upper)
            self.processing.jaws_X_pix[0][2, 1] = (
                self.processing.isocenters_pix[0][4, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][5, 0]

        # Set distance between pelvis-abdomen isocenters to have symmetric fields
        if self.processing.isocenters_pix[0][2, 2] < x_iliac:
            # Thorax isocenters
            self.processing.isocenters_pix[0][4, 2] = (
                self.processing.isocenters_pix[0][3, 2]
                + self.processing.isocenters_pix[0][6, 2]
            ) / 2.1
            self.processing.isocenters_pix[0][5, 2] = (
                self.processing.isocenters_pix[0][3, 2]
                + self.processing.isocenters_pix[0][6, 2]
            ) / 2.1

            # Adjust fields of abdomen and thorax iso to the positions of ribs and iliac crests
            self.processing.jaws_X_pix[0][2, 1] = (
                self.optimization_result.x_pixel_ribs
                - self.processing.isocenters_pix[0][2, 2]
            ) * self.aspect_ratio

            self.processing.jaws_X_pix[0][5, 0] = (
                self.optimization_result.x_pixel_iliac
                - self.processing.isocenters_pix[0][4, 2]
            ) * self.aspect_ratio

            # Fix overlap of thorax field (upper)
            self.processing.jaws_X_pix[0][4, 1] = (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][4, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]

            # Field overlap pelvis-abdomen isocenters
            x_midpoint_pelvis_abdomen = (
                self.processing.isocenters_pix[0][0, 2]
                + self.processing.isocenters_pix[0][2, 2]
            ) / 2
            self.processing.jaws_X_pix[0][0, 1] = (
                x_midpoint_pelvis_abdomen
                - self.processing.isocenters_pix[0][0, 2]
                + self.field_overlap_pixels / 2
            ) * self.aspect_ratio
            self.processing.jaws_X_pix[0][3, 0] = (
                -(
                    x_midpoint_pelvis_abdomen
                    - self.processing.isocenters_pix[0][0, 2]
                    + self.field_overlap_pixels / 2
                )
                * self.aspect_ratio
            )

        self._fit_collimator_pelvic_field()

    def _adjust_field_geometry_arms(self) -> None:
        """Adjust the field geometry predicted by the arms model, according to the maximum extension
        of iliac crests and ribs."""

        x_iliac = (
            self.optimization_result.x_pixel_iliac
        )  # iliac crests 'x' pixel location
        x_ribs = self.optimization_result.x_pixel_ribs  # ribs pixel 'x' location

        if (
            x_ribs - (x_ribs - x_iliac) / 2
            < self.processing.isocenters_pix[0][2, 2]
            < x_ribs + (x_ribs - x_iliac)
            or self.processing.isocenters_pix[0][2, 2] < x_iliac
        ):
            # Setting the isocenters for arms model at 3/4 space
            self.processing.isocenters_pix[0][2, 2] = (
                x_iliac + (x_ribs - x_iliac) * 3 / 4
            )
            self.processing.isocenters_pix[0][3, 2] = (
                x_iliac + (x_ribs - x_iliac) * 3 / 4
            )
            # Adjust the fields (abdomen/pelvis) after the isocenter shift, with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][2, 1] = (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]
            self.processing.jaws_X_pix[0][0, 1] = (
                self.processing.isocenters_pix[0][2, 2]
                - self.processing.isocenters_pix[0][0, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][3, 0]

        # Isocenter on the spine
        if x_iliac < self.processing.isocenters_pix[0][2, 2] < x_ribs:
            self.processing.jaws_X_pix[0][2, 0] = (
                (x_iliac - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )
            self.processing.jaws_X_pix[0][3, 1] = (
                (x_ribs - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )

            # Ensure overlap of abdominal field (lower) and pelvic field (upper), with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][3, 0] = -(
                (
                    self.processing.isocenters_pix[0][2, 2]
                    - self.processing.isocenters_pix[0][0, 2]
                    + self.field_overlap_pixels
                )
                * self.aspect_ratio
                - self.processing.jaws_X_pix[0][0, 1]
            )

            # Ensure overlap of abdominal field (upper) and thorax field (lower), with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][2, 1] = (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]

        # Adjust the fields (abdomen/pelvis) according to iliac crests and ribs location
        if self.processing.isocenters_pix[0][2, 2] > x_ribs:
            self.processing.jaws_X_pix[0][0, 1] = (
                x_ribs - self.processing.isocenters_pix[0][0, 2]
            ) * self.aspect_ratio
            self.processing.jaws_X_pix[0][3, 0] = (
                x_iliac - self.processing.isocenters_pix[0][2, 2]
            ) * self.aspect_ratio

        # Fix overlap of abdomen field (upper)
        self.processing.jaws_X_pix[0][2, 1] = (
            self.processing.isocenters_pix[0][6, 2]
            - self.processing.isocenters_pix[0][2, 2]
            + self.field_overlap_pixels
        ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]

        self._fit_collimator_pelvic_field()

    def _adjust_field_geometry_arms_5_355(self) -> None:
        x_iliac = (
            self.optimization_result.x_pixel_iliac
        )  # iliac crests 'x' pixel location
        x_ribs = self.optimization_result.x_pixel_ribs  # ribs pixel 'x' location

        if (
            x_ribs - (x_ribs - x_iliac) / 2
            < self.processing.isocenters_pix[0][2, 2]
            < x_ribs + (x_ribs - x_iliac)
            or self.processing.isocenters_pix[0][2, 2] < x_iliac
            or self.processing.isocenters_pix[0][2, 2] > x_ribs
        ):
            # Setting the isocenters for arms model at 3/4 space
            self.processing.isocenters_pix[0][2, 2] = (
                x_iliac + (x_ribs - x_iliac) * 3 / 4
            )
            self.processing.isocenters_pix[0][3, 2] = (
                x_iliac + (x_ribs - x_iliac) * 3 / 4
            )
            # Adjust the fields (abdomen/pelvis) after the isocenter shift, with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][2, 1] = (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]

        # Isocenter on the spine
        if x_iliac < self.processing.isocenters_pix[0][2, 2] < x_ribs:
            self.processing.jaws_X_pix[0][2, 0] = (
                (x_iliac - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )
            self.processing.jaws_X_pix[0][3, 1] = (
                (x_ribs - self.processing.isocenters_pix[0][2, 2])
                * self.aspect_ratio
                / 2
            )
            # Ensure overlap of abdominal field (lower) and pelvic field (upper), with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][3, 0] = -(
                (
                    self.processing.isocenters_pix[0][2, 2]
                    - self.processing.isocenters_pix[0][0, 2]
                    + self.field_overlap_pixels
                    - self.processing.jaws_Y_pix[0][0, 1]
                )
                * self.aspect_ratio
            )

            # Ensure overlap of abdominal field (upper) and thorax field (lower), with an overlap specified in config.yml
            self.processing.jaws_X_pix[0][2, 1] = (
                self.processing.isocenters_pix[0][6, 2]
                - self.processing.isocenters_pix[0][2, 2]
                + self.field_overlap_pixels
            ) * self.aspect_ratio + self.processing.jaws_X_pix[0][7, 0]

    def _fit_collimator_pelvic_field(self):
        ptv_mask = self.processing.masks[0][..., 1]
        y_pixels = np.arange(
            round(
                self.processing.isocenters_pix[0][0, 0]
                + self.processing.jaws_Y_pix[0][1, 0] * self.aspect_ratio
            ),
            round(
                self.processing.isocenters_pix[0][0, 0]
                + self.processing.jaws_Y_pix[0][1, 1] * self.aspect_ratio
            ),
            1,
            dtype=int,
        )
        x_lower_field = round(self.processing.isocenters_pix[0][0, 2])
        search_space = {
            "x_lowest": np.arange(
                0,
                x_lower_field,
                1,
                dtype=int,
            )
        }

        def _loss(pos_new):
            # Maximize the ptv field coverage while minimizing the field aperture
            x_lowest = pos_new["x_lowest"]
            return np.count_nonzero(ptv_mask[y_pixels, x_lowest:x_lower_field] != 0) - (
                x_lower_field - x_lowest
            )

        opt = GridSearchOptimizer(search_space)
        opt.search(
            _loss,
            n_iter=search_space["x_lowest"].size,
            verbosity=False,
        )

        self.processing.jaws_X_pix[0][1, 0] = (
            opt.best_value[0] - self.processing.isocenters_pix[0][0, 2] - 3
        ) * self.aspect_ratio

    def _define_search_space(self):
        """Define the optimization search space: 'x' pixel boundary and 'y' pixels range."""
        self.optimization_search_space.x_pixel_left = round(
            (
                self.processing.isocenters_pix[0][0, 2]
                + self.processing.isocenters_pix[0][2, 2]
            )
            / 2
        )
        if MODEL == "body":
            self.optimization_search_space.x_pixel_left += 10
        else:
            self.optimization_search_space.x_pixel_left -= 10

        self.optimization_search_space.x_pixel_right = round(
            (
                self.processing.isocenters_pix[0][2, 2]
                + self.processing.isocenters_pix[0][6, 2]
            )
            / 2
        )

        x_com = round(
            ndimage.center_of_mass(self.processing.masks[0][..., 0])[
                0
            ]  # pyright: ignore[reportArgumentType]
        )
        self.optimization_search_space.y_pixels_right = np.arange(
            x_com - 115, x_com - 50
        )
        self.optimization_search_space.y_pixels_left = np.arange(
            x_com + 50, x_com + 115
        )

    def _search_iliac_and_ribs(self):
        """Search the optimal 'x' pixel location of the iliac crests and ribs."""
        ptv_mask = self.processing.masks[0][..., 1]

        self._define_search_space()

        search_space = {
            "x_iliac": np.arange(
                self.optimization_search_space.x_pixel_left,
                self.optimization_search_space.x_pixel_right,
                1,
                dtype=int,
            ),
            "x_ribs": np.arange(
                self.optimization_search_space.x_pixel_left,
                self.optimization_search_space.x_pixel_right,
                1,
                dtype=int,
            ),
        }

        best_value_ribs = self.original_size
        best_value_iliac = 0
        for y_pixels in (
            self.optimization_search_space.y_pixels_right,
            self.optimization_search_space.y_pixels_left,
        ):

            def _loss(pos_new):
                """
                Compute the optimization loss based on the given position.

                Parameters:
                - pos_new (dict): Dictionary containing the positions of iliac crests and ribs.
                                    It has the keys "x_iliac" and "x_ribs" representing the pixel
                                    locations of the iliac crests and ribs.

                Returns:
                - score (float): The computed optimization score, which is a combination of two terms:
                                1. Maximizing background pixels while minimizing pixels in the mask.
                                2. Maximizing the count of background pixels along the 'y' pixels for a
                                    given candidate 'x' pixel location.
                """
                x_iliac = pos_new["x_iliac"]
                x_ribs = pos_new["x_ribs"]

                # pylint: disable=cell-var-from-loop
                score = 2 * (
                    np.count_nonzero(ptv_mask[y_pixels, x_iliac:x_ribs] == 0)
                    - np.count_nonzero(ptv_mask[y_pixels, x_iliac:x_ribs] != 0)
                ) + 60 * (
                    np.count_nonzero(ptv_mask[y_pixels, x_ribs] == 0)
                    + np.count_nonzero(ptv_mask[y_pixels, x_iliac] == 0)
                )
                # pylint: enable=cell-var-from-loop

                return score

            def _constraint_x_pixel(pos_new):
                return pos_new["x_iliac"] <= pos_new["x_ribs"]

            opt = ParallelTemperingOptimizer(
                search_space, constraints=[_constraint_x_pixel], population=20
            )
            opt.search(
                _loss,
                n_iter=1000,
                verbosity=False,
            )

            if opt.best_value[1] < best_value_ribs:
                best_value_ribs = opt.best_value[1]

            if best_value_iliac < opt.best_value[0]:
                best_value_iliac = opt.best_value[0]

        if best_value_ribs < best_value_iliac:
            print(
                "Pixel location of ribs < iliac crests. Local optimization might be incorrect."
            )

        self.optimization_result.x_pixel_iliac = best_value_iliac
        self.optimization_result.x_pixel_ribs = best_value_ribs

    def optimize(self) -> None:
        """
        Search the 'x' pixel coordinates of the maximum extension
        of the ribs and iliac crests and optimize the abdominal field geometry.
        """

        try:
            assert self.processing.masks[0].shape == (
                512,
                self.original_size,
                3,
            )
        except AssertionError:
            print(
                "Expected original shape image. The local optimization might give incorrect results."
            )

        # Maximum distance between head-pelvis isocenters: 840 mm
        maximum_extension_pix = 840 / self.slice_thickness
        head_pelvis_iso_pix_diff = (
            self.processing.isocenters_pix[0][8, 2]  # head iso-z
            - self.processing.isocenters_pix[0][0, 2]  # pelvis iso-z
        )
        head_pelvis_iso_dist_pix = abs(head_pelvis_iso_pix_diff)

        if head_pelvis_iso_dist_pix > maximum_extension_pix:
            shift_pixels = np.sign(head_pelvis_iso_pix_diff) * (
                head_pelvis_iso_dist_pix - maximum_extension_pix
            )
            print(
                f"Distance between head-pelvis isocenters was {int(head_pelvis_iso_dist_pix)} pixels."
                f" Maximum allowed distance is {int(maximum_extension_pix)} (= 84 cm)."
                f" Shifting pelvis isocenters by {int(shift_pixels)} pixels.",
            )
            self.processing.isocenters_pix[0][[0, 1], 2] = (
                self.processing.isocenters_pix[0][0, 2] + shift_pixels
            )

        self._search_iliac_and_ribs()
        if MODEL == "body":
            if COLL_5_355:
                self._adjust_field_geometry_body_5_355()
            else:
                self._adjust_field_geometry_body()
        elif MODEL == "arms":
            if COLL_5_355:
                self._adjust_field_geometry_arms_5_355()
            else:
                self._adjust_field_geometry_arms()
