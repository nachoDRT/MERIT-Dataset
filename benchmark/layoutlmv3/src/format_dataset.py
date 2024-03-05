import json
import os
import random
from typing import List
import shutil
from glob import glob


DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

# Folders where data is originally stored
IMAGES_DIR = "/app/data/dataset_output/images/"
ANNOTATIONS_DIR = "/app/data/dataset_output/annotations/"


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
    split_index = int(len(file_names) * fractions[0])
    train_files = file_names[:split_index]
    eval_files = file_names[split_index:]
    files = [train_files, eval_files]

    for files_partition, partition in zip(files, partitions):
        move_files(files_partition, "".join([partition, "_data"]))


def create_zip():
    zip_this = "/app/data/dataset/"
    save_here = "/app/data/dataset/"
    shutil.make_archive(base_name=save_here, format="zip", root_dir=zip_this)


if __name__ == "__main__":

    d_features = read_dataset_features_json()
    partitions = [part for part in d_features["dataset_partitions"].keys()]
    partitions_fractions = [frac for frac in d_features["dataset_partitions"].values()]

    split_dataset(partitions, partitions_fractions)
    create_zip()
