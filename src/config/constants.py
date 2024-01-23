import json
from os.path import dirname, join

# Get the correct file path depending on the working directory
config_dir_path = dirname(__file__)
constants_json_path = join(config_dir_path, "constants.json")

with open(constants_json_path) as file:
    data = json.load(file)

PTV_EXCLUDE_SUBSTRINGS: list[str] = data["ptv_exclude_substrings"]
MAP_ID_PTV: dict[str, str] = data["map_id_ptv"]
MAP_ID_JUNCTION: dict[str, None | str | list[str]] = data["map_id_junction"]
DICOM_PATH: str = data["dicom_path"]
DICOM_DIR_TEST_PATH: str = data["dicom_dir_test_path"]
MAP_ID_LUNGS: dict[str, list[str]] = data["map_lung_roi_name"]
MODEL: str = data["model"]
COLL_5_355: bool = data["coll_5_355"]

project_dir_path = dirname(dirname(config_dir_path))
if COLL_5_355:
    dir_path = join(project_dir_path, "data", "5_355")
else:
    dir_path = join(project_dir_path, "data", "90")
RAW_DATA_DIR_PATH: str = join(dir_path, "raw")
INTERIM_DATA_DIR_PATH: str = join(dir_path, "interim")
