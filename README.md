# tmi-isocenter

A deep learning model that predicts the **field geometry** of Total Marrow Irradiation (TMI) radiotherapy plans: where to place the isocenters and how wide to open the X/Y jaws of each treatment field.

- **Input:** a 2D coronal projection of the patient CT (3 channels: CT density inside the target, target mask, organs at risk).
- **Output:** a vector of isocenter coordinates and jaw apertures, learned by a convolutional neural network with a regression head.
- **Stack:** PyTorch + PyTorch Lightning, TensorBoard for monitoring.

> **No installation option:** you can run the whole project in your browser, on a free GPU, with Google Colab. See the [Colab guide](notebooks/colab/COLAB.md).
> The rest of this README explains how to run the project on your own computer.

## 1. Requirements

- **Python 3.11.1** (this exact version was used to develop and test the project; other 3.11.x releases should also work, newer versions are **not** supported by the pinned dependencies).
- Git.
- About 3 GB of free disk space for the virtual environment.
- A GPU is optional. Training runs on the CPU by default.

## 2. Set up the project

### 2.1 Get the code

```
git clone <repository-url> tmi-isocenter
cd tmi-isocenter
git checkout daihs
```

### 2.2 Install Python 3.11.1

Pick **one** of the options below.

**Option A: python.org installer**

Download and install Python 3.11.1 from <https://www.python.org/downloads/release/python-3111/>.
On Windows, the installer adds the `py` launcher, so you can check the installation with:

```
py -3.11 --version
```

On macOS/Linux check it with `python3.11 --version`.

**Option B: uv (no admin rights needed)**

[uv](https://docs.astral.sh/uv/) downloads and manages Python versions for you. Install it following the instructions on its website, then run:

```
uv python install 3.11.1
```

### 2.3 Create and activate the virtual environment

Run these commands from the repository root folder.

| | Create the venv | Activate it |
|---|---|---|
| Windows (cmd.exe) | `py -3.11 -m venv venv` | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `py -3.11 -m venv venv` | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `python3.11 -m venv venv` | `source venv/bin/activate` |
| Any OS, with uv | `uv venv venv --python 3.11.1 --seed` | as above for your OS |

> PowerShell may refuse to run the activation script. If so, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd.exe instead.

Once activated, your prompt starts with `(venv)`. Check the version:

```
python --version      # must print Python 3.11.1
```

**Always activate the venv before running any command below.**

### 2.4 Install the dependencies

```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The last command installs the project itself as a package, so that the `src.` imports work from scripts and notebooks.

*Optional, NVIDIA GPU users:* the default PyTorch wheel on Windows is CPU-only. To train on a GPU, replace it with the CUDA build:

```
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### 2.5 Create the configuration file

The code reads its settings from `src/config/constants.json`, which is not tracked by git. Create it from the template:

```
# Windows
copy src\config\constants.example.json src\config\constants.json
# macOS / Linux
cp src/config/constants.example.json src/config/constants.json
```

The settings you will want to change are:

| Key | Values | Meaning |
|---|---|---|
| `model` | `"body"`, `"arms"` | `body` trains on patients **without** isocenters on the arms, `arms` on patients **with** isocenters on the arms. |
| `coll_5_355` | `true` | Collimator angle of the pelvis fields (5°/355°). Keep it `true`: the shared dataset contains only this configuration (`data/5_355`). |
| `num_workers` | `0`, `2`, `4`, ... | Number of processes used to load data. `0` is the safest choice, especially on Windows. |

The other keys (`dicom_path`, `map_id_*`, ...) are only used to extract the dataset from the original DICOM files and can be left as they are.

### 2.6 Add the dataset

The patient data is **not** part of the repository. Copy the `5_355` folder you received from the instructor into a `data` folder in the repository root, so that the layout is:

```
tmi-isocenter/
└── data/
    └── 5_355/
        ├── patient_info.csv
        └── raw/
            ├── ptv_imgs2D.npz        # CT density inside the target (one 2D image per patient)
            ├── ptv_masks2D.npz       # target (PTV) masks
            ├── brain_masks2D.npz     # organs at risk masks
            ├── lungs_masks2D.npz
            ├── liver_masks2D.npz
            ├── intestine_masks2D.npz
            ├── bladder_masks2D.npz
            ├── isocenters_pix.npy    # labels: isocenters of the 12 fields, in pixels
            ├── jaws_X_pix.npy        # labels: X jaw apertures
            ├── jaws_Y_pix.npy        # labels: Y jaw apertures
            └── angles.npy            # collimator angles
```

The `data/` folder is ignored by git: never commit patient data.

## 3. Run the project

All commands are run from the repository root, with the venv activated.

### Step 1: Pre-process the data

```
python -m src.data.processing
```

This combines the masks into 3-channel images, resizes them to 512×512 and applies the same transformation to the labels. The result is saved in `data/5_355/interim/`. You only need to run it once (and again if you change the processing code).

### Step 2: Train the model

```
python -m src.train_cnn
```

This trains the model selected by `model` in `constants.json` (up to 50 epochs, with early stopping on the validation loss) and then evaluates it on the test set. On a CPU, expect about 11 minutes per epoch: for a shorter experiment, limit the number of epochs:

```
python -m src.train_cnn --max-epochs 5
```

Everything is saved in `lightning_logs/<model>_model_5_355/version_<N>/`:
- `checkpoints/*.ckpt`: the trained weights.
- `img/`: plots of the predicted vs. ground-truth fields on the test patients.

### Step 3: Monitor training with TensorBoard

In a second terminal (venv activated):

```
tensorboard --logdir lightning_logs
```

Then open <http://localhost:6006> in your browser to follow losses and learning rate.

### Step 4: Predict with a trained model

Open [notebooks/prediction.ipynb](notebooks/prediction.ipynb) (for example in VS Code with the Python and Jupyter extensions, selecting the `venv` interpreter as kernel). Set `ckpt_path` to one of your checkpoints, e.g.:

```python
ckpt_path = r"..\lightning_logs\body_model_5_355\version_0\checkpoints\<file>.ckpt"
```

and run all cells. The predicted field geometries are plotted and saved in `notebooks/img/`.

[notebooks/visualize.ipynb](notebooks/visualize.ipynb) shows how the data looks. The cells in the *Visualize all PTV's* section work with the shared dataset, while the cells above need the original DICOM files and can be skipped.

## 4. Project structure

```
src/
├── config/        # constants.py loads constants.json and derives paths / output size
├── data/
│   ├── dicom_preprocess.py   # DICOM -> data/<coll>/raw  (needs the original DICOMs, already done)
│   ├── processing.py         # raw -> interim (resize, scale, rotate)
│   ├── dataset*.py           # train/val/test split, augmentation, target vector
│   └── augmentation.py
├── models/cnn.py             # the CNN architecture
├── modules/                  # Lightning modules (loss, training/validation/test steps)
├── utils/                    # coordinate transforms, plotting, post-processing optimization
└── train_cnn.py              # training entry point
notebooks/                    # prediction and data visualization
```

Some ideas to play with: the CNN architecture in `src/models/cnn.py`, the loss weights and learning rate in `src/modules/`, the data augmentation in `src/data/augmentation.py`, and the batch size / epochs / early stopping in `src/train_cnn.py`.

## 5. Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`:** run `pip install -e .` with the venv activated, and launch scripts with `python -m ...` from the repository root.
- **`FileNotFoundError: ...constants.json`:** you skipped step 2.5.
- **`FileNotFoundError: ...interim/masks2D.npy`:** run Step 1 before training.
- **Pip fails to install the requirements:** check that `python --version` prints 3.11.x inside the venv.

## For contributors

To enable the formatting hooks (black, isort, nbstripout, ...) before committing:

```
pre-commit install
```
