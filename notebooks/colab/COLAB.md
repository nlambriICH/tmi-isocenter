# Run the project in Google Colab

With Colab you can run the whole project in your browser, on a free GPU, without installing anything on your computer. You only need a Google account.

There are two notebooks:

| Notebook | What it does | Open |
|---|---|---|
| `tmi-isocenter-training.ipynb` | Trains the model and saves it to your Google Drive | [**Open in Colab**](https://colab.research.google.com/github/nlambriICH/tmi-isocenter/blob/daihs/notebooks/colab/tmi-isocenter-training.ipynb) |
| `tmi-isocenter-evaluation.ipynb` | Loads a trained model and evaluates it on the test patients | [**Open in Colab**](https://colab.research.google.com/github/nlambriICH/tmi-isocenter/blob/daihs/notebooks/colab/tmi-isocenter-evaluation.ipynb) |

After opening a notebook, click **Copy to Drive** in the toolbar. You get your own copy, so you can save your changes.

## How to run a notebook

You need the **dataset passkey**, which the instructor gives you. The dataset is encrypted: without the passkey it cannot be opened. Do not write the passkey in the notebook or share it.

1. Menu **Runtime → Run all**.
2. The first cell installs `conda` and **restarts the session**. Colab may show a *"session crashed"* or *"restarting"* message: this is expected.
3. Run **Runtime → Run all** a second time. The installation is skipped and the rest of the notebook runs.
4. When the notebook asks for the **Dataset passkey**, type it in the box and press Enter. It is asked once per session.

Both notebooks first download the code and the dataset, then create a Python 3.11 environment (3-5 min, only the first time in a session).

## 1. Train the model

Use a GPU: menu **Runtime → Change runtime type → T4 GPU → Save**. The notebook also runs without a GPU, but training is about 5× slower. The free GPU is not always available: if Colab says so, try again later or continue on CPU with fewer epochs.

| Step | Time on T4 GPU | Time on CPU |
|---|---|---|
| Setup (first run only) | ~5 min | ~5 min |
| One training epoch | ~2 min | ~11 min |
| Test + plots | ~1 min | ~1 min |

The **Settings** cell, right after the first cell, controls the run:

- `MODEL`: `"body"` (patients without isocenters on the arms) or `"arms"` (patients with isocenters on the arms).
- `MAX_EPOCHS`: maximum number of training epochs. The full training uses 50; start with 10.
- `N_PLOTS`: number of test predictions to display.

After changing them, click the Settings cell and use **Runtime → Run cell and below**. The setup is not repeated.

To change the code, open the file browser (folder icon on the left), go to `tmi-isocenter/src`, double-click a file to edit it, then re-run the training cell. Your edits are lost when the session ends, so copy anything you want to keep.

**Save your model:** Colab deletes all files when the session ends (after a period of inactivity, or at most 12 hours). Run **Step 7** to copy the trained model, TensorBoard logs and plots to `MyDrive/tmi-isocenter-results/<model>_version_<N>` in your Google Drive.

## 2. Evaluate a trained model

No GPU needed: the default CPU runtime is fine. The evaluation takes about a minute after the setup.

By default the notebook takes the **most recent run** saved in `MyDrive/tmi-isocenter-results` and guesses the model type from the folder name. In the **Settings** cell you can instead:

- set `CHECKPOINT_PATH` to a specific `.ckpt` file, e.g. an older run on Drive (`/content/drive/MyDrive/tmi-isocenter-results/body_version_0/checkpoints/<file>.ckpt`) or a file you uploaded with the file browser;
- set `MODEL` to `"body"` or `"arms"` if it cannot be guessed. It must match the model type used for training.

The notebook shows the error (MSE) of each of the 11 test patients and, for the best ones, the predicted fields next to the real plan. Step 8 saves the plots and the metrics table to Drive, inside the run folder.

## Troubleshooting

- **Strange errors after the instructor updated the code:** start from a clean machine with **Runtime → Disconnect and delete runtime**, then run the notebook again. The download step prints the code version it is using.
- **The session restarts in the middle of the notebook:** run *Runtime → Run all* again. Completed steps are skipped.
- **`Wrong passkey` error:** re-run the cell and type the passkey again (check spaces and upper/lower case). The dataset is not downloaded again.
- **`Missing data files` error or a download error:** the dataset download failed (Google Drive limits downloads). Wait a few minutes and re-run the cell.
- **`GPU available: False` in the training notebook:** you are on a CPU runtime. Change it and run the notebook again (the session restarts and setup is repeated).
- **TensorBoard does not show:** it is optional. Skip that cell; the results are also shown in Step 6.
- **"You cannot currently connect to a GPU due to usage limits":** Colab's free GPU quota is used up. Use a CPU runtime with a small `MAX_EPOCHS`, or try again later.
- **`No saved runs` in the evaluation notebook:** run Step 7 of the training notebook first, with the same Google account.
- **`The checkpoint predicts ... values` error:** `MODEL` does not match the trained model. Set it to the other type (`"body"` or `"arms"`).

## For instructors: update the dataset

The notebooks download an AES-256 encrypted archive of `data/5_355` (anonymized `patient_info.csv` + `raw/`) from Google Drive.

1. From the repository root, run `bash scripts/encrypt_dataset.sh` and choose a passkey (at least 12 characters). It creates `tmi-dataset-5_355.tar.gz.enc`, which is ignored by git.
2. Upload the file to Google Drive and share it as *Anyone with the link*. Copy the file ID from the link (`https://drive.google.com/file/d/<FILE_ID>/view`).
3. Set `DATA_FILE_ID` to that ID in the dataset cell of both notebooks, then commit and push.
4. Give the passkey to the students outside the repository. To change the passkey, repeat steps 1-4.
