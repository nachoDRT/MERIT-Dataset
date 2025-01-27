from os.path import abspath, dirname, join
from typing import Dict, List
from datasets import Dataset, Features, Image, Value, DatasetDict
from huggingface_hub import HfApi, Repository, HfFolder
from utils import *
from io import BytesIO


def format_data(data: Dict) -> DatasetDict:
    # Convert to HF dataset
    features = Features({"image": Image(), "ground_truth": Value("string")})
    train_dataset = Dataset.from_dict(data["train"], features=features)
    test_dataset = Dataset.from_dict(data["test"], features=features)
    validation_dataset = Dataset.from_dict(data["validation"], features=features)

    # Create DatasetDict
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": validation_dataset})

    return dataset


def push_splits_to_hf(data: DatasetDict, configuration: Dict, subset: str, repo_name: str = "merit"):
    # Authentication
    HfFolder.save_token(configuration["hf_token"])

    api = HfApi()
    user = api.whoami()
    username = user["name"]

    for split, dataset in data.items():
        dataset_name = f"{username}/{repo_name}"
        dataset.push_to_hub(dataset_name, config_name=subset, split=split)


def get_hf_config() -> Dict:

    configuration_path = join(dirname(dirname(abspath(__file__))), "config", "hf_config.json")
    configuration = read_json(configuration_path)

    return configuration


def push_dataset_to_hf(dataset, subset: str):

    hf_config = get_hf_config()
    push_splits_to_hf(dataset, hf_config, subset)


if __name__ == "__main__":

    # push_dataset_to_hf()
    pass
