"""Script to evaluate a trained model on the test patients"""

import argparse
import csv
import os
from os.path import join

import torch
from torch.utils.data import DataLoader

from src.config.constants import MODEL, OUTPUT_DIM
from src.utils.visualization_cnn import Visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint on the test patients."
    )
    parser.add_argument(
        "--ckpt", required=True, help="Path to the model checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation",
        help="Folder where the plots and the metrics are saved.",
    )
    args = parser.parse_args()

    if MODEL == "arms":
        from src.data.dataset_arms import DatasetArms as DatasetClass
        from src.modules.arms_cnn import ArmsCNN as ModuleClass
    elif MODEL == "body":
        from src.data.dataset_body import DatasetBody as DatasetClass
        from src.modules.body_cnn import BodyCNN as ModuleClass
    else:
        from src.data.dataset import Dataset as DatasetClass
        from src.modules.lightning_cnn import LitCNN as ModuleClass

    model = ModuleClass.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    ckpt_output_dim = model.hparams["output"]
    if ckpt_output_dim != OUTPUT_DIM:
        raise ValueError(
            f"The checkpoint predicts {ckpt_output_dim} values, but model={MODEL!r} in "
            f"constants.json expects {OUTPUT_DIM}. Set 'model' to the type of the "
            "trained model."
        )

    dataset = DatasetClass()
    test_dataset = dataset.prediction_dataset()  # fixed test split (random_state=42)
    df_patient_info = dataset.df_patient_info_original

    os.makedirs(args.output_dir, exist_ok=True)
    viz = Visualize()
    results = []

    with torch.no_grad():
        for x, y_reg, patient_idx in DataLoader(test_dataset):
            y_reg_hat = model(x)
            mse = model.weighted_mse_loss(y_reg_hat, y_reg.view(1, -1))
            idx = int(patient_idx.item())
            patient_id = df_patient_info.iloc[idx]["PatientID"]

            # Saves img/test/predicted/test_<idx>.png and img/test/ground_truth/test_<idx>.png
            viz.plot_img(
                input_img=x.numpy()[0],
                patient_idx=idx,
                output=y_reg_hat[0],
                path=args.output_dir,
                mse=mse,
            )
            results.append(
                {"patient_idx": idx, "patient_id": patient_id, "mse": mse.item()}
            )
            print(f"Patient {patient_id} (index {idx}): MSE = {mse.item():.6f}")

    metrics_path = join(args.output_dir, "test_metrics.csv")
    with open(metrics_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["patient_idx", "patient_id", "mse"])
        writer.writeheader()
        writer.writerows(results)

    mean_mse = sum(r["mse"] for r in results) / len(results)
    print(f"\nTest patients: {len(results)} | mean MSE = {mean_mse:.6f}")
    print(f"Plots and metrics saved in {os.path.abspath(args.output_dir)}")
