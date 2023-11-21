from pathlib import Path
from typing import List
from os.path import dirname, abspath
import os
import json
import copy


REQS_PATH = os.path.join(
    Path(dirname(abspath(__file__))).parent,
    "replication_pipeline",
    "assets",
    "requirements_v2.json",
)


def get_available_schools_per_language():
    """
    Get the name of every school with a template available in src/replication_pipeline/
    templates/

    Returns:
        dict: A dictionary with languages as keys and lists of string (school names) as
              values.

    """
    templates_path = os.path.join(
        Path(__file__).resolve().parents[1], "replication_pipeline", "templates"
    )

    available_langs = {}
    for i, (root, dirs, _) in enumerate(os.walk(templates_path)):
        for dir in dirs:
            if i == 0:
                available_langs[dir] = []
            try:
                available_langs[Path(root).parts[-1]].append(dir)

            except KeyError:
                pass

    return available_langs


def read_json(file_path: str) -> dict:
    """
    Read a JSON file from a given path.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def save_json(path: str, data_content: dict):
    """
    Save a JSON file.

    Args:
        path (str): The path where to save the file (including the .json extension).

    """
    with open(path, "w") as file:
        json.dump(data_content, file, indent=4)


def update_schools_requirements(selected_schools: List):
    """
    Update the schools to replicate in the requirements JSON.

    Args:
        selected_schools (List): A list of strings with the names of those schools that
                                 the user wants to replicate.

    """
    previous_data = read_json(REQS_PATH)
    updated_data = copy.deepcopy(previous_data)
    languages_data = updated_data["samples"]

    for language_data in languages_data.values():
        for school_data in language_data.values():
            school_data["include"] = False
            for school in selected_schools:
                if school_data["nickname"] == school:
                    school_data["include"] = True

    save_json(REQS_PATH, updated_data)


def update_fe_male_proportion_requirements(team_a_percentage: int):
    """
    Update the female/male students proportion in the requirements JSON.

    Args:
        team_a_percentage (int): A number 0-100 that indicates the percentage of
                                'team_a' (female in this case) proportion the user wants
                                to include in the dataset.

    """

    previous_data = read_json(REQS_PATH)
    updated_data = copy.deepcopy(previous_data)
    updated_data["female_proportion"] = round(team_a_percentage / 100, 2)
    updated_data["male_proportion"] = round(1 - updated_data["female_proportion"], 2)
    save_json(REQS_PATH, updated_data)


if __name__ == "__main__":
    pass
