import json
import os
from tqdm import tqdm
import random
import string
from typing import List
import shutil
from glob import glob


DATASET_FEATURES_JSON = "/app/config/dataset_features.json"

# Folders where data is originally stored
IMAGES_DIR = "/app/data/dataset_output/images/"
ANNOTATIONS_DIR = "/app/data/dataset_output/annotations/"


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


def move_files(file_names, split):
    # Create the destination folders
    imgs_path = os.path.join("/app/data/dataset", split)
    os.makedirs(imgs_path, exist_ok=True)
    annoations_path = os.path.join("/app/data/dataset_output", split, "annotations")
    print("CAMBIOS AQUI", annoations_path)
    os.makedirs(annoations_path, exist_ok=True)

    for name in file_names:
        image_src = os.path.join(IMAGES_DIR, name + ".png")
        annotation_src = os.path.join(ANNOTATIONS_DIR, name + ".json")

        image_dst = os.path.join(imgs_path, name + ".png")
        annotation_dst = os.path.join(annoations_path, name + ".json")

        shutil.move(image_src, image_dst)
        shutil.move(annotation_src, annotation_dst)


def split_dataset(partitions: List, fractions: List, lang: str):
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
        move_files(files_partition, "".join([lang, "_", partition]))


def create_zip():
    zip_this = "/app/data/dataset/"
    save_here = "/app/data/dataset"
    shutil.make_archive(base_name=save_here, format="zip", root_dir=zip_this)


if __name__ == "__main__":

    d_features = read_datset_features_json()
    partitions = [part for part in d_features["dataset_partitions"].keys()]
    partitions_fractions = [frac for frac in d_features["dataset_partitions"].values()]
    language = d_features["dataset_language"]

    split_dataset(partitions, partitions_fractions, language)

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
