import json
import os
import random
import string
import shutil
import argparse
from typing import List
from pathlib import Path
from glob import glob
from tqdm import tqdm

LANGUAGE = "spanish"
DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

# Folders where data is stored after gathering
ROOT = "/app/data/"
IMGS_DIR_SUFIX = "/images/"
ANNOTATIONS_DIR_SUFIX = "/annotations/"
ASSERT_TOLERANCE = 1e-3


def generate_json(*, json_path: str, dataset_partition: str, dict: dict):

    json_object = json.dumps(dict, indent=4)
    json_name_no_json_extension = os.path.join(
        json_path, "".join([dataset_partition, "_json"])
    )
    json_name = "".join([json_name_no_json_extension, ".json"])

    with open(json_name, "w") as outfile:
        outfile.write(json_object)

    os.rename(json_name, json_name_no_json_extension)


def get_random_string():

    length = 10
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def read_datset_features_json():

    with open(DATASET_FEATURES_JSON, "r") as dataset_features:
        data = dataset_features.read()
        d_features = json.loads(data)

    return d_features


def process_data(dicts_list: List, file_path: str):
    for file in tqdm(sorted(os.listdir(file_path))):
        if file.endswith(".json"):

            with open(os.path.join(file_path, file), "r") as openfile:
                json_object = json.load(openfile)

            interior = {}
            fname = "".join([file[:-5], ".png"])
            interior["fname"] = fname
            interior["width"] = 1654
            interior["height"] = 2339

            intermediate = {}
            intermediate["id"] = file[:-5]
            intermediate["uid"] = get_random_string()
            intermediate["document"] = json_object["form"]
            intermediate["img"] = interior
            dicts_list.append(intermediate)

    return dicts_list


def copy_files(file_names, lang, split):
    # Create the destination folders
    split = "".join([lang, "_", split])
    imgs_path = os.path.join("/app/data/dataset", split)

    try:
        shutil.rmtree(imgs_path)
    except FileNotFoundError:
        pass
    os.makedirs(imgs_path, exist_ok=True)

    annoations_path = os.path.join("/app/data/dataset_output", split, "annotations")
    try:
        shutil.rmtree(annoations_path)
    except FileNotFoundError:
        pass
    os.makedirs(annoations_path, exist_ok=True)

    if split == "".join([lang, "_train"]) or split == "".join([lang, "_eval"]):
        imgs_dir = "".join([ROOT, "train-val", IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, "train-val", ANNOTATIONS_DIR_SUFIX])
    else:
        imgs_dir = "".join([ROOT, "test", IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, "test", ANNOTATIONS_DIR_SUFIX])

    for name in file_names:
        image_src = os.path.join(imgs_dir, name + ".png")
        annotation_src = os.path.join(annotations_dir, name + ".json")

        image_dst = os.path.join(imgs_path, name + ".png")
        annotation_dst = os.path.join(annoations_path, name + ".json")

        shutil.copy(image_src, image_dst)
        shutil.copy(annotation_src, annotation_dst)


def split_dataset(partitions: List, fractions: List, lang: str, test_data: bool = None):
    train_validation_dir = "".join([ROOT, "train-val", IMGS_DIR_SUFIX, "*.png"])
    # Obtain the file names
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
        copy_files(files_partition, lang, partition)


def create_zip():
    zip_this = "/app/data/dataset/"
    save_here = "/app/data/dataset"
    shutil.make_archive(base_name=save_here, format="zip", root_dir=zip_this)


def gather_files(gathering_paths: list):
    """Gather all the files (stored by language and school) in a common place"""

    # Loop over partitions (train/val and test)
    for partition in os.listdir(ROOT):

        if partition in gathering_paths:
            images_dir = "".join([ROOT, partition, IMGS_DIR_SUFIX])
            annotations_dir = "".join([ROOT, partition, ANNOTATIONS_DIR_SUFIX])

            # Make sure there is nothing from previous iteration
            try:
                shutil.rmtree(images_dir)
                shutil.rmtree(annotations_dir)
            except FileNotFoundError:
                pass
            # Folders to place gathered data
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(annotations_dir, exist_ok=True)

            partition_path = os.path.join(ROOT, partition)

            # Loop over languages
            for language in os.listdir(partition_path):
                if language == LANGUAGE:
                    language_path = os.path.join(partition_path, language)

                    # Loop over schools
                    for school in os.listdir(language_path):
                        school_path = os.path.join(language_path, school)
                        annotations_path = os.path.join(
                            school_path, "dataset_output", "annotations"
                        )
                        images_path = os.path.join(
                            school_path, "dataset_output", "images"
                        )

                        # Gather annotations
                        print(
                            f"{partition.upper()}: Gathering annotations for {school} in {language}"
                        )
                        for file in tqdm(os.listdir(annotations_path)):
                            source_file = os.path.join(annotations_path, file)
                            dest_file = os.path.join(annotations_dir, file)
                            shutil.copy(source_file, dest_file)

                        # Gather images
                        print(f"Gathering images for {school} in {language}")
                        for file in tqdm(os.listdir(images_path)):
                            source_file = os.path.join(images_path, file)
                            dest_file = os.path.join(images_dir, file)
                            shutil.copy(source_file, dest_file)
                else:
                    pass
        else:
            pass


def get_partitions_and_fractions(d_features: dict):
    partitions = [part for part in d_features["dataset_partitions"].keys()]
    partitions_fractions = [frac for frac in d_features["dataset_partitions"].values()]

    # In case there there is a specific test set, increase the training dataset fraction
    if sum(partitions_fractions) + ASSERT_TOLERANCE < 1:
        partitions_fractions[0] += 1 - sum(partitions_fractions)

    return partitions, partitions_fractions


def get_where_to_gather(gathering_paths: list) -> list:
    folders = []

    for path in gathering_paths:
        folders.append(Path(path).name)

    return folders


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_folder", type=str)
    parser.add_argument("--gather_train_val_data_from", type=str)
    parser.add_argument("--gather_test_data_from", type=str)
    args = parser.parse_args()

    # Get where the user wants to get the data from
    gathering_folders = get_where_to_gather(
        [args.gather_train_val_data_from, args.gather_test_data_from]
    )

    # Dataset features
    d_features = read_datset_features_json()

    if eval(args.test_data_folder) != None:
        d_features["dataset_partitions"].pop("test")

    partitions, partitions_fractions = get_partitions_and_fractions(d_features)
    language = d_features["dataset_language"]

    # Files are firstly arranged by lang and school. We collect them in a common place
    gather_files(gathering_folders)
    # Split files in train, validation and test partitions
    split_dataset(
        partitions, partitions_fractions, language, eval(args.test_data_folder)
    )

    for partition in partitions:
        partiton_name = "".join([language, "_", partition])
        file_path = os.path.join(
            "/app/data/dataset_output", partiton_name, "annotations"
        )

        new_json = {}
        new_json["lang"] = language
        new_json["version"] = "0.1"
        new_json["split"] = partition

        dicts_list = []

        if os.path.isdir(file_path):
            process_data(dicts_list, file_path)

        new_json["documents"] = dicts_list
        generate_json(
            json_path="/app/data/dataset",
            dataset_partition=partiton_name,
            dict=new_json,
        )

    create_zip()
