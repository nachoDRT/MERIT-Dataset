import json
import os
import random
from typing import List
import shutil
from glob import glob
from tqdm import tqdm

LANGUAGE = "english"
DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

# Folders where data is originally stored
IMAGES_DIR = "/app/data/dataset_output/images/"
ANNOTATIONS_DIR = "/app/data/dataset_output/annotations/"
ASSERT_TOLERANCE = 1e-3

GATHER_FILES_PATH = "/app/data/original/"


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

    for name in file_names:
        image_src = os.path.join(IMAGES_DIR, name + ".png")
        annotation_src = os.path.join(ANNOTATIONS_DIR, name + ".json")

        image_dst = os.path.join(imgs_path, name + ".png")
        annotation_dst = os.path.join(annoations_path, name + ".json")

        shutil.move(image_src, image_dst)
        shutil.move(annotation_src, annotation_dst)


def split_dataset(partitions: List, fractions: List):
    # Obtain the file names
    file_names = [
        os.path.splitext(os.path.basename(x))[0]
        for x in glob(os.path.join(IMAGES_DIR, "*.png"))
    ]
    random.shuffle(file_names)

    # Split train and eval files
    assert (
        abs(sum(fractions) - 1) < ASSERT_TOLERANCE
    ), f"Your dataset fractions might not be right: {fractions}. Please check"

    split_index_train = int(len(file_names) * fractions[0])
    split_index_eval = split_index_train + int(len(file_names) * fractions[1])

    # Split train, validation and test files
    train_files = file_names[:split_index_train]
    eval_files = file_names[split_index_train:split_index_eval]
    test_files = file_names[split_index_eval:]

    files = [train_files, eval_files, test_files]


def create_zip():
    zip_this = "/app/data/dataset/"
    save_here = "/app/data/dataset/"
    shutil.make_archive(base_name=save_here, format="zip", root_dir=zip_this)


def gather_files():
    """Gather all the files (stored by language and school) in a common place"""

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    # Loop over languages
    for language in tqdm(os.listdir(GATHER_FILES_PATH)):

        if language == LANGUAGE:
            language_path = os.path.join(GATHER_FILES_PATH, language)

            # Loop over schools
            for school in tqdm(os.listdir(language_path)):
                print(f"Gathering samples for {school} in {language}")
                school_path = os.path.join(language_path, school)
                annotations_path = os.path.join(
                    school_path, "dataset_output", "annotations"
                )
                images_path = os.path.join(school_path, "dataset_output", "images")

                # Gather annotations
                for archivo in tqdm(os.listdir(annotations_path)):
                    source_file = os.path.join(annotations_path, archivo)
                    dest_file = os.path.join(ANNOTATIONS_DIR, archivo)
                    shutil.move(source_file, dest_file)

                # Gather images
                for archivo in tqdm(os.listdir(images_path)):
                    source_file = os.path.join(images_path, archivo)
                    dest_file = os.path.join(IMAGES_DIR, archivo)
                    shutil.move(source_file, dest_file)
        else:
            pass


if __name__ == "__main__":

    d_features = read_dataset_features_json()
    partitions = [part for part in d_features["dataset_partitions"].keys()]
    partitions_fractions = [frac for frac in d_features["dataset_partitions"].values()]

    # Files are firstly arranged by lang and school. We collect them in a common place
    gather_files()
    # Split files in train, validation and test partitions
    split_dataset(partitions, partitions_fractions)
    create_zip()
