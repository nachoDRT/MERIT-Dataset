import json
import os
import random
from typing import List
import shutil
from glob import glob
from tqdm import tqdm
import argparse

LANGUAGE = "english"
DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

GATHER_FILES_PATH = "/app/data/"

# Folders where data is stored after gathering
# IMAGES_DIR = "/app/data/dataset_output/images/"
# ANNOTATIONS_DIR = "/app/data/dataset_output/annotations/"
ASSERT_TOLERANCE = 1e-3

# Folders where data is stored after gathering
ROOT = "/app/data/"
IMGS_DIR_SUFIX = "/images/"
ANNOTATIONS_DIR_SUFIX = "/annotations/"


def read_dataset_features_json():

    with open(DATASET_FEATURES_JSON, "r") as dataset_features:
        data = dataset_features.read()
        d_features = json.loads(data)

    return d_features


def move_files(file_names, split):
    # Create the destination folders
    imgs_path = os.path.join("/app/data/dataset", split, "images")
    os.makedirs(imgs_path, exist_ok=True)
    annoations_path = os.path.join("/app/data/dataset", split, "annotations")
    os.makedirs(annoations_path, exist_ok=True)

    if split == "training_data" or split == "validating_data":
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

        shutil.move(image_src, image_dst)
        shutil.move(annotation_src, annotation_dst)


def split_dataset(partitions: List, fractions: List, test_data: bool = None):
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
        partitions.append("testing")

    else:
        test_files = file_names[split_index_eval:]

    files = [train_files, eval_files, test_files]

    for files_partition, partition in zip(files, partitions):
        move_files(files_partition, "".join([partition, "_data"]))


def create_zip():
    zip_this = "/app/data/dataset/"
    save_here = "/app/data/dataset/"
    shutil.make_archive(base_name=save_here, format="zip", root_dir=zip_this)


def gather_files():
    """Gather all the files (stored by language and school) in a common place"""

    # Loop over partitions (train/val and test)
    for partition in tqdm(os.listdir(GATHER_FILES_PATH)):

        # Folders to place gathered data
        # os.makedirs(IMAGES_DIR, exist_ok=True)
        # os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

        images_dir = "".join([ROOT, partition, IMGS_DIR_SUFIX])
        annotations_dir = "".join([ROOT, partition, ANNOTATIONS_DIR_SUFIX])

        # Folders to place gathered data
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)

        partition_path = os.path.join(GATHER_FILES_PATH, partition)

        # Loop over languages
        for language in tqdm(os.listdir(partition_path)):
            if language == LANGUAGE:
                language_path = os.path.join(partition_path, language)

                # Loop over schools
                for school in tqdm(os.listdir(language_path)):
                    print(f"Gathering samples for {school} in {language}")
                    school_path = os.path.join(language_path, school)
                    annotations_path = os.path.join(
                        school_path, "dataset_output", "annotations"
                    )
                    images_path = os.path.join(school_path, "dataset_output", "images")

                    # Gather annotations
                    for file in tqdm(os.listdir(annotations_path)):
                        source_file = os.path.join(annotations_path, file)
                        dest_file = os.path.join(annotations_dir, file)
                        shutil.move(source_file, dest_file)

                    # Gather images
                    for file in tqdm(os.listdir(images_path)):
                        source_file = os.path.join(images_path, file)
                        dest_file = os.path.join(images_dir, file)
                        shutil.move(source_file, dest_file)
            else:
                pass


def get_partitions_and_fractions(d_features: dict):
    partitions = [part for part in d_features["dataset_partitions"].keys()]
    partitions_fractions = [frac for frac in d_features["dataset_partitions"].values()]

    # In case there there is a specific test set, increase the training dataset fraction
    if sum(partitions_fractions) + ASSERT_TOLERANCE < 1:
        partitions_fractions[0] += 1 - sum(partitions_fractions)

    return partitions, partitions_fractions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_folder", type=str)
    args = parser.parse_args()

    d_features = read_dataset_features_json()

    if eval(args.test_data_folder) != None:
        d_features["dataset_partitions"].pop("testing")

    partitions, partitions_fractions = get_partitions_and_fractions(d_features)
    # Files are firstly arragned by lang and school. We collect them in a common place
    gather_files()
    # Split files in train, validation and test partitions
    split_dataset(partitions, partitions_fractions, eval(args.test_data_folder))
    create_zip()
