import json

with open(r"src\config\constants.json", "r") as file:
    data = json.load(file)

PTV_EXCLUDE_SUBSTRINGS: list[str] = data["ptv_exclude_substrings"]
MAP_ID_PTV: dict[str, str] = data["map_id_ptv"]
MAP_ID_JUNCTION: dict[str, str] = data["map_id_junction"]
DICOM_PATH: str = data["dicom_path"]
