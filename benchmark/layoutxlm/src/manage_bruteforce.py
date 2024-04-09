import subprocess
import itertools
import argparse
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).parents[1]


def get_dataset_language() -> str:
    dataset_config_path = os.path.join(ROOT, "config", "dataset_features.json")
    dataset_config = read_json(dataset_config_path)

    dataset_lang_abbreviation = dataset_config["dataset_language"]
    dataset_language = dataset_config["xlm_languages_map"][dataset_lang_abbreviation]

    return dataset_language


def read_json(path: str) -> dict:

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def get_available_subsets(paths_map: dict) -> list:
    storing_path = paths_map["storing_path"]

    subsets = []
    for file in tqdm(sorted(os.listdir(storing_path))):
        subsets.append(file)

    return subsets


def get_subsets_combinations(subsets: list, n: int = 2):
    combinations = list(itertools.combinations(subsets, n))

    return combinations


def get_paths() -> dict:
    relevant_paths = {}

    dataset_language = get_dataset_language()

    # The folder where the combinations must be stored to launch a training session
    target_path = os.path.join(ROOT, "data", "train-val", dataset_language)
    relevant_paths["target_path"] = target_path

    # The folder where the subsets are stored
    storing_path = os.path.join(ROOT, "data", "bruteforce", dataset_language)
    relevant_paths["storing_path"] = storing_path

    return relevant_paths


def check_elements_in_folder(comb: list, target_path: str) -> list:
    checked_comb = []

    for element in comb:
        if element not in os.listdir(target_path):
            checked_comb.append(element)

    return checked_comb


def move_in_data(comb: tuple, paths_map: dict):
    target_path = paths_map["target_path"]
    storing_path = paths_map["storing_path"]

    # Check if any of the combination elements is inside the target path and update comb
    comb = check_elements_in_folder(list(comb), target_path)

    for element in comb:
        src_path = os.path.join(storing_path, element)
        dst_path = os.path.join(target_path, element)

        shutil.move(src_path, dst_path)


def move_out_data(comb: tuple, combinations: list, index: int, paths_map: dict):
    target_path = paths_map["target_path"]
    storing_path = paths_map["storing_path"]

    move_out = []

    try:
        next_comb = combinations[index + 1]
        for element in comb:
            if element not in next_comb:
                move_out.append(element)

        for element in move_out:
            src_path = os.path.join(target_path, element)
            dst_path = os.path.join(storing_path, element)

            shutil.move(src_path, dst_path)

    except IndexError:
        pass


def check_elements_in_target_folder(n: int, paths_map: dict):
    target_path = paths_map["target_path"]

    for element in os.listdir(target_path):
        print(f"{element} is in the target folder")

    if len(os.listdir(target_path)) != n:
        raise Warning(f"There are more elements in the target folder than expected")


def manage_training_session(combs: list, paths_map: dict, n: int):
    for i, combination in enumerate(combs):
        print(combination)
        move_in_data(combination, paths_map)
        check_elements_in_target_folder(n, paths_map)
        # Launch data processing and training
        move_out_data(combination, combinations, i, paths_map)
        print("")


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--combinations_length", type=int)
    args = parser.parse_args()

    # Get parsed values
    n = args.combinations_length

    # Get dataset target and storing paths
    relevant_paths = get_paths()

    # Get all the subsets (school templates) and get all possible combinations
    subsets = get_available_subsets(relevant_paths)
    combinations = get_subsets_combinations(subsets, n)

    manage_training_session(combinations, relevant_paths, n)
