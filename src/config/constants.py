import os
import json

# Get the correct file path depending on the working directory
config_dir_path = os.path.dirname(__file__)
constants_json_path = os.path.join(config_dir_path, r"constants.json")

with open(constants_json_path, "r") as file:
    data = json.load(file)

PTV_EXCLUDE_SUBSTRINGS: list[str] = data["ptv_exclude_substrings"]
MAP_ID_PTV: dict[str, str] = data["map_id_ptv"]
MAP_ID_JUNCTION: dict[str, str] = data["map_id_junction"]
DICOM_PATH: str = data["dicom_path"]
DICOM_TEST_PATH: str = data["dicom_test_path"]
