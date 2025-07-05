import debugpy
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict


def download_dataset(name: str) -> DatasetDict:
    print("Downloading dataset")
    data = load_dataset(name)

    return data


def create_folder(folder: str, silent_mode: bool = True) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)
        if not silent_mode:
            print(f"Folder '{folder}' created")


def inspect_dataset(data: DatasetDict, root: str) -> None:

    for partition_key, partition_values in data.items():
        # Create a partition folder
        partition_path = os.path.join(root, partition_key)
        create_folder(partition_path, False)
        print(f"Processing {partition_key.capitalize()}")

        for i, sample in tqdm(enumerate(partition_values)):
            sample_id = str(i).zfill(4)
            sample_path = os.path.join(partition_path, sample_id)
            create_folder(sample_path)

            # Get and save annotations
            annotations_path = os.path.join(sample_path, "".join([sample_id, ".json"]))
            annotations = sample["ground_truth"]
            annotations = process_annotations(annotations)
            save_annotations(annotations, annotations_path)

            # Get and save image
            img_path = os.path.join(sample_path, "".join([sample_id, ".png"]))
            image = sample["image"]
            image.save(img_path)


def process_annotations(annotations_str: str) -> json:
    annotations_json = json.loads(annotations_str)
    return annotations_json


def save_annotations(annoatations_json: json, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annoatations_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to connect...")
    debugpy.wait_for_client()

    # Load dataset
    dataset_name = "naver-clova-ix/cord-v2"
    dataset = download_dataset(dataset_name)

    # Files comunication pipeline
    volume_folder = "/app/dataset_inspection"
    create_folder(volume_folder)

    # Check what the dataset contains
    inspect_dataset(dataset, volume_folder)
