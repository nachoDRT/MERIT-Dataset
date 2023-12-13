from pathlib import Path
from typing import List, Dict
from os.path import dirname, abspath
import os
import json
import copy

ASSETS_PATH = os.path.join(
    Path(dirname(abspath(__file__))).parent, "replication_pipeline", "assets"
)

REQS_PATH = os.path.join(ASSETS_PATH, "requirements_v2.json")


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


def update_fe_male_bias_distributions(*args):
    previous_data = read_json(REQS_PATH)
    updated_data = copy.deepcopy(previous_data)

    groups = ["female", "male"]

    # Assuming a normal distribution
    for group_i, index in enumerate(range(0, len(args), 2)):
        avrg = args[index]
        dev = args[index + 1]

        bias_data = {}
        bias_data["average"] = avrg
        bias_data["deviation"] = dev
        updated_data[f"{groups[group_i]}_bias_distribution"] = bias_data

    save_json(REQS_PATH, updated_data)


def update_ethnic_origins_in_requirements(
    proportions: List, origins: Dict, language: str
):
    previous_data = read_json(REQS_PATH)
    updated_data = copy.deepcopy(previous_data)
    if previous_data != None:
        previous_data_origins = previous_data["origins"]

    if origins != None:
        origins_for_given_language = origins[language]

    mapped_info = {}

    for i, proportion in enumerate(proportions):
        if i == 0:
            mapped_info[language] = round(proportion, 0)
        else:
            try:
                mapped_info[origins_for_given_language[i - 1].capitalize()] = round(
                    proportion, 0
                )
            except IndexError:
                pass

    if origins != None:
        for lang_i in origins:
            # We do not want to loose previous data
            if previous_data_origins != None:
                try:
                    if type(previous_data_origins[lang_i]) == dict:
                        origins[lang_i] = previous_data_origins[lang_i]
                except KeyError:
                    pass
        origins[language] = mapped_info

    updated_data["origins"] = origins

    save_json(REQS_PATH, updated_data)


def get_ethnic_origins_proportions(lang: str):
    data = read_json(REQS_PATH)

    proportions = []
    try:
        proportions_dict = data["origins"][lang]

        for element in proportions_dict:
            proportions.append(proportions_dict[element])

        return proportions

    except (KeyError, TypeError) as e:
        return proportions


def props_2_values(proportions):
    values = []
    accumulated = 0
    for i, value in enumerate(proportions):
        if i + 1 == len(proportions):
            break

        accumulated += value
        values.append(accumulated)

    return values


def get_langs_with_replicas():
    """
    Get those languages containing schools the user wants to replicate.

    Returns:
        List: A list of strings with the languages of interest
    """

    langs_of_interest = []
    data = read_json(REQS_PATH)

    languages_data = data["samples"]

    for lang, language_data in languages_data.items():
        for school_data in language_data.values():
            if school_data["include"] == True:
                langs_of_interest.append(lang)
                break

    return langs_of_interest


def get_available_origin_langs():
    available_origin_langs = []

    files_to_check = {}
    files_to_check["familiy"] = False
    files_to_check["female"] = False
    files_to_check["male"] = False

    for root, _, files in os.walk(ASSETS_PATH):
        candidate = Path(root).parts[-2]
        for file in files:
            family_file = f"family_names_{candidate}.txt"
            females_file = f"female_first_names_{candidate}.txt"
            males_file = f"male_first_names_{candidate}.txt"

            if file == family_file:
                files_to_check["familiy"] = True
            elif file == females_file:
                files_to_check["female"] = True
            elif file == males_file:
                files_to_check["male"] = True

        if all(files_to_check.values()):
            available_origin_langs.append(candidate)
            for key in files_to_check:
                files_to_check[key] = False

    available_origin_langs = sorted(list(set(available_origin_langs)))

    return available_origin_langs


if __name__ == "__main__":
    pass
