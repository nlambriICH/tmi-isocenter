import json
from os.path import dirname, join

# Get the correct file path depending on the working directory
config_dir_path = dirname(__file__)
constants_json_path = join(config_dir_path, "constants.json")

with open(constants_json_path) as file:
    data = json.load(file)

PTV_EXCLUDE_SUBSTRINGS: list[str] = data["ptv_exclude_substrings"]
DICOM_PATH: str = data["dicom_path"]
DICOM_DIR_TEST_PATH: str = data["dicom_dir_test_path"]
MAP_ID_LUNGS: dict[str, list[str]] = data["map_lung_roi_name"]
MODEL: str = data["model"]
COLL_5_355: bool = data["coll_5_355"]
NUM_WORKERS: int = data["num_workers"]  # Number of processes used by DataLoader

project_dir_path = dirname(dirname(config_dir_path))
if COLL_5_355:
    dir_path = join(project_dir_path, "data", "5_355")
    map_id_ptv = "map_id_ptv_coll_5_355"
    map_id_junction = "map_id_junction_coll_5_355"
    if data["model"] == "body":
        output = 19
    else:
        output = 24
else:
    dir_path = join(project_dir_path, "data", "90")
    map_id_ptv = "map_id_ptv_coll_90"
    map_id_junction = "map_id_junction_coll_90"
    if data["model"] == "body":
        output = 25
    else:
        output = 30

MAP_ID_PTV: dict[str, str] = data[map_id_ptv]
MAP_ID_JUNCTION: dict[str, None | str | list[str]] = data[map_id_junction]
RAW_DATA_DIR_PATH: str = join(dir_path, "raw")
INTERIM_DATA_DIR_PATH: str = join(dir_path, "interim")
OUTPUT_DIM: int = output  # Dimensionality of CNN Regressin Head
