import json
import os
import random
import argparse
import shutil
import pandas as pd
from typing import List, Dict
from glob import glob
from tqdm import tqdm
from pathlib import Path

DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

GATHER_FILES_PATH = "/app/data/"

ASSERT_TOLERANCE = 1e-3

# Folders where data is stored after gathering
ROOT = "/app/data/"
IMGS_DIR_SUFIX = "/images/"
ANNOTATIONS_DIR_SUFIX = "/annotations/"

MAX_WAVINESS = 4.5


def read_json(json_path: str) -> Dict:
    with open(json_path, "r") as json_content:
        data = json_content.read()
        data = json.loads(data)
    return data


def read_dataset_features_json():
    dataset_features = read_json(DATASET_FEATURES_JSON)
    return dataset_features


def get_blueprint():
    """
    Retrieves the dataset blueprint from a CSV file.

    This method builds the path to the CSV blueprint ('dataset_blueprint.csv'),
    located in the 'dashboard' directory. It then reads the CSV file into a pandas
    DataFrame.

    Returns:
        tuple: A tuple containing two elements:
            - pandas.DataFrame: The DataFrame created from the CSV file.
            - str: The file path to the CSV file.
    """

    blueprint_path = "/app/config/dataset_blueprint.csv"
    blueprint_df = pd.read_csv(blueprint_path)

    return blueprint_df, blueprint_path


def move_files(file_names, split):
    # Create the destination folder
    dst_path = os.path.join("/app/output", split)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    if split == "train" or split == "validation":
        imgs_dir = "".join([ROOT, "train-val", IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, "train-val", ANNOTATIONS_DIR_SUFIX])
    else:
        imgs_dir = "".join([ROOT, "test", IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, "test", ANNOTATIONS_DIR_SUFIX])

    for name in file_names:
        image_src = os.path.join(imgs_dir, name + ".png")
        annotation_src = os.path.join(annotations_dir, name + ".json")

        image_dst = os.path.join(dst_path, name + ".png")
        annotation_dst = os.path.join(dst_path, name + ".json")

        shutil.move(image_src, image_dst)
        shutil.move(annotation_src, annotation_dst)


def split_dataset(partitions: List, fractions: List, test_data: bool = None):
    print("\nSplitting Dataset\n")
    # Obtain the file names
    train_validation_dir = "".join([ROOT, "train-val", IMGS_DIR_SUFIX, "*.png"])
    file_names = [
        os.path.splitext(os.path.basename(x))[0] for x in glob(train_validation_dir)
    ]
    random.shuffle(file_names)

    assert (
        abs(sum(fractions) - 1) < ASSERT_TOLERANCE
    ), f"Your dataset fractions might not be right: {fractions}. Please check"

    split_index_train = int(len(file_names) * fractions[0])
    split_index_eval = split_index_train + int(len(file_names) * fractions[1])

    # Split train, validation and test files
    train_files = file_names[:split_index_train]
    eval_files = file_names[split_index_train:split_index_eval]

    if test_data:
        test_files_path = "".join([ROOT, "test", IMGS_DIR_SUFIX, "*.png"])
        test_files = [
            os.path.splitext(os.path.basename(x))[0] for x in glob(test_files_path)
        ]
        random.shuffle(test_files)
        partitions.append("test")

    else:
        test_files = file_names[split_index_eval:]

    files = [train_files, eval_files, test_files]

    for files_partition, partition in zip(files, partitions):
        print(f"{partition.upper()} SAMPLES: {len(files_partition)}")
        move_files(files_partition, partition)


def check_file_name(name: str) -> str:
    flag = name[-1].isdigit()
    if flag:
        return name
    else:
        return name[:-12]


def gather_files(data_language: str, format: str, dataset_features: Dict):
    """Gather all the files (stored by language and school) in a common place"""

    # Read the blueprint and convert waviness values to float
    blueprint_df, _ = get_blueprint()
    blueprint_df["waviness"] = (
        blueprint_df["waviness"].str.replace(",", ".").astype(float)
    )

    # Loop over partitions (train/val and test)
    for partition in tqdm(os.listdir(GATHER_FILES_PATH)):

        images_dir = "".join([ROOT, partition, IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, partition, ANNOTATIONS_DIR_SUFIX])

        # Folders to place gathered data
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        partition_path = os.path.join(GATHER_FILES_PATH, partition)

        # Loop over languages
        for language in tqdm(os.listdir(partition_path)):
            if language == data_language:
                language_path = os.path.join(partition_path, language)

                # Loop over schools
                for school in tqdm(os.listdir(language_path)):
                    print(
                        f"Gathering samples for {school.capitalize()} in {language.capitalize()}"
                    )
                    school_path = os.path.join(language_path, school)
                    annotations_path = os.path.join(
                        school_path, "dataset_output", "annotations"
                    )
                    images_path = os.path.join(school_path, "dataset_output", "images")

                    # Gather annotations
                    for file in tqdm(os.listdir(annotations_path)):
                        file_name = file.split(".")[0]
                        file_name = check_file_name(file_name)
                        waviness_value = float(
                            blueprint_df.loc[
                                blueprint_df["file_name"] == file_name, "waviness"
                            ].iloc[0]
                        )
                        words_out = blueprint_df.loc[
                            blueprint_df["file_name"] == file_name, "words_out"
                        ].iloc[0]
                        source_file = os.path.join(annotations_path, file)
                        dest_file = os.path.join(annotations_dir, file)

                        if format == "cord-v2":
                            source_formatted_file = os.path.join(
                                annotations_path, "".join([file_name, "_cord-v2_.json"])
                            )
                            format_annotations_cordv2_style(
                                source_file,
                                source_formatted_file,
                                dataset_features["years"],
                            )
                            source_file = source_formatted_file

                        if waviness_value < MAX_WAVINESS and not words_out:
                            shutil.move(source_file, dest_file)

                    # Gather images
                    for file in tqdm(os.listdir(images_path)):
                        file_name = file.split(".")[0]
                        file_name = check_file_name(file_name)
                        waviness_value = float(
                            blueprint_df.loc[
                                blueprint_df["file_name"] == file_name, "waviness"
                            ].iloc[0]
                        )
                        words_out = blueprint_df.loc[
                            blueprint_df["file_name"] == file_name, "words_out"
                        ].iloc[0]
                        source_file = os.path.join(images_path, file)
                        dest_file = os.path.join(images_dir, file)
                        if waviness_value < MAX_WAVINESS and not words_out:
                            shutil.move(source_file, dest_file)
            else:
                pass


def process_extractions(years: List, subjects: List, grades: Dict) -> Dict:
    ground_truth = {}

    for year in years:
        ground_truth_year = []
        for subject in subjects:
            subject_dict = {}
            subject_dict["subject"] = list(subject.values())[0]
            subject_dict["grade"] = grades[list(subject.keys())[0]]
            ground_truth_year.append(subject_dict)
        ground_truth[year] = ground_truth_year

    return ground_truth


def extract_key_annotations(dataset_data: Dict, academic_years: List):
    years = []
    subjects = []
    grades = {}

    for segment in dataset_data["form"]:
        if segment["label"] == "other":
            pass

        # Look for academic years
        elif segment["label"] in academic_years:
            years.append(segment["label"])

        # Look for grades
        elif segment["label"].split("_")[-1] == "answer":
            grade = segment["text"]
            subject = segment["label"][:-7]
            grades[subject] = grade

        # Otherwise it is a subject
        else:
            subject_dict = {}
            subject_dict[segment["label"]] = segment["text"]
            subjects.append(subject_dict)

    ground_truth = process_extractions(years, subjects, grades)

    return ground_truth


def format_annotations_cordv2_style(
    source: str, formatted: str, academic_years: List
) -> None:

    funsd_format_data = read_json(source)
    cordv2_format_data = {}

    cordv2_format_data = extract_key_annotations(funsd_format_data, academic_years)
    cordv2_format_data = {"gt_parse": cordv2_format_data}

    with open(formatted, "w") as output_file:
        json.dump(cordv2_format_data, output_file, indent=4)


def get_partitions_and_fractions(dataset_features: dict):
    partitions = [part for part in dataset_features["dataset_partitions"].keys()]
    partitions_fractions = [
        frac for frac in dataset_features["dataset_partitions"].values()
    ]

    # In case there is a specific test set, increase the training dataset fraction
    if sum(partitions_fractions) + ASSERT_TOLERANCE < 1:
        partitions_fractions[0] += 1 - sum(partitions_fractions)

    return partitions, partitions_fractions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_folder", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--annotations_format", type=str)
    args = parser.parse_args()

    d_features = read_dataset_features_json()

    if eval(args.test_data_folder) != None:
        d_features["dataset_partitions"].pop("test")

    partitions, partitions_fractions = get_partitions_and_fractions(d_features)
    # Files are firstly arragned by lang and school. We collect them in a common place
    gather_files(args.language, args.annotations_format, d_features)
    # Split files in train, validation and test partitions
    split_dataset(partitions, partitions_fractions, eval(args.test_data_folder))
