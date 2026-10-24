# Run the project in Google Colab

With Colab you can run the whole project in your browser, on a free GPU, without installing anything on your computer. You only need a Google account.

## 1. Open the notebook

Click here: [**Open tmi-isocenter-colab.ipynb in Colab**](https://colab.research.google.com/github/nlambriICH/tmi-isocenter/blob/daihs/notebooks/tmi-isocenter-colab.ipynb)

Then click **Copy to Drive** in the toolbar. You get your own copy, so you can save your changes.

## 2. Select a GPU

Menu **Runtime → Change runtime type → T4 GPU → Save**.

The notebook also runs without a GPU, but training is about 5× slower. The free GPU is not always available: if Colab says so, try again later or continue on CPU with fewer epochs.

## 3. Run the notebook

1. Menu **Runtime → Run all**.
2. The first cell installs `conda` and **restarts the session**. Colab may show a *"session crashed"* or *"restarting"* message: this is expected.
3. Run **Runtime → Run all** a second time. The installation is skipped and the remaining steps run in order:
   - download the code and the dataset,
   - create a Python 3.11 environment and install the dependencies (~3-5 min),
   - pre-process the data,
   - train and test the model,
   - show the predictions on the test patients.

| Step | Time on T4 GPU | Time on CPU |
|---|---|---|
| Setup (first run only) | ~5 min | ~5 min |
| One training epoch | ~2 min | ~11 min |
| Test + plots | ~1 min | ~1 min |

## 4. Experiment

The **Settings** cell, right after the first cell, controls the run:

- `MODEL`: `"body"` (patients without isocenters on the arms) or `"arms"` (patients with isocenters on the arms).
- `MAX_EPOCHS`: maximum number of training epochs. The full training uses 50; start with 10.
- `N_PLOTS`: number of test predictions to display.

After changing them, click the Settings cell and use **Runtime → Run cell and below**. The setup is not repeated.

To change the code, open the file browser (folder icon on the left), go to `tmi-isocenter/src`, double-click a file to edit it, then re-run the training cell. Your edits are lost when the session ends, so copy anything you want to keep.

## 5. Save your results

Colab deletes all files when the session ends (after a period of inactivity, or at most 12 hours). Run **Step 7** of the notebook to copy the trained model, TensorBoard logs and plots to `MyDrive/tmi-isocenter-results` in your Google Drive.

## Troubleshooting

- **Strange errors after the instructor updated the code:** start from a clean machine with **Runtime → Disconnect and delete runtime**, then run the notebook again. Step 2 prints the code version it is using.
- **The session restarts in the middle of the notebook:** run *Runtime → Run all* again. Completed steps are skipped.
- **`Missing data files` error:** the dataset download failed (Google Drive limits downloads). Wait a few minutes and re-run the cell.
- **`GPU available: False`:** you are on a CPU runtime. Change it as in step 2 and run the notebook again (the session restarts and setup is repeated).
- **TensorBoard does not show:** it is optional. Skip that cell; the results are also shown in Step 6.
- **"You cannot currently connect to a GPU due to usage limits":** Colab's free GPU quota is used up. Use a CPU runtime with a small `MAX_EPOCHS`, or try again later.
