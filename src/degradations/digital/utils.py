from os.path import join, abspath, dirname
from typing import List
import os


def get_merit_subset_paths(language: str, school: str) -> List:
    root_path = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "data")
    subset_path = join(root_path, "original", language, school, "dataset_output", "annotations")

    file_list = []
    if os.path.exists(subset_path):
        file_list = [join(subset_path, f) for f in os.listdir(subset_path) if os.path.isfile(join(subset_path, f))]

    return file_list
