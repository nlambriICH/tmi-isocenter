import os
from src.config.constants import COLL_5_355


def coll_dir() -> str:
    if COLL_5_355:
        dir_path = os.path.abspath(r"data/5_355/")
    else:
        dir_path = os.path.abspath(r"data/90/")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def directory(in_path: str):
    dir_path = coll_dir()
    return os.path.abspath(os.path.join(dir_path, in_path))
