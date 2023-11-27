"""Lightning module for CNN training"""
from src.modules.lightning_cnn import LitCNN


class ArmCNN(LitCNN):  # pylint: disable=too-many-ancestors
    """Lightning module for CNN training"""

    def __init__(
        self,
        learning_rate=1e-5,
        mse_loss_weight=5.0,
        bcelogits_loss_weight=0.0000001,
        weight=3,
        focus_on=[0, 1],
        filters=4,
        output=30,
    ):
        """Initialize the LitCNN module

        Args:
            cnn (torch.nn.Module): CNN module with multi-head output for keypoints regression
                and angle classification
        """
        super().__init__(
            learning_rate=learning_rate,
            mse_loss_weight=mse_loss_weight,
            bcelogits_loss_weight=bcelogits_loss_weight,
            weight=weight,
            focus_on=focus_on,
            filters=filters,
            output=output,
        )
