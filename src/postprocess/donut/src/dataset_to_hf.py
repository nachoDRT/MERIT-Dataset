import os
import json
from typing import Dict
from datasets import Dataset, Features, Image, Value, DatasetDict
from huggingface_hub import HfApi, Repository, HfFolder

DATASET_NAME = "cord-target"


def load_data(data_dir: str) -> Dict:
    images = []
    ground_truths = []

    # Iterate over files
    for fname in os.listdir(data_dir):
        if fname.endswith(".png"):
            img_path = os.path.join(data_dir, fname)
            json_path = os.path.join(data_dir, fname.replace('.png', '.json'))

            # Read annotations
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            
            # Join image and annotations
            images.append(img_path)
            ground_truths.append(json.dumps(annotations))
    
    return {"image": images, "ground_truth": ground_truths}


def format_data(data: Dict) -> DatasetDict:
     # Convert to HF dataset
    features = Features({
        "image": Image(),
        "ground_truth": Value("string")
    })
    train_dataset = Dataset.from_dict(data["train"], features=features)
    test_dataset = Dataset.from_dict(data["test"], features=features)
    validation_dataset = Dataset.from_dict(data["validation"], features=features)

    # Create DatasetDict
    dataset = DatasetDict({"train": train_dataset, 
                           "test": test_dataset, 
                           "validation": validation_dataset})
    
    return dataset


def push_dataset_to_hf(data: DatasetDict, configuration: Dict):
    # Authentication
    HfFolder.save_token(configuration["hf_token"])

    api = HfApi()
    user = api.whoami()
    username = user['name']

    repo_name = "dummy-cord"

    for split, dataset in data.items():
        dataset_name = f"{username}/{repo_name}"
        dataset.push_to_hub(dataset_name, split=split)


def get_hf_config() -> Dict:
    with open("/app/config/hf_config.json") as f:
        configuration = json.load(f)
    
    return configuration


if __name__ ==  "__main__":

    # Get HuggingFace config
    hf_config = get_hf_config()
    
    # Load data (imgs + annotations)
    root = os.path.join("/app", DATASET_NAME)
    
    train_data = load_data(os.path.join(root, "train"))
    validation_data = load_data(os.path.join(root, "validation"))
    test_data = load_data(os.path.join(root, "test"))
    
    dataset = {"train": train_data, "test": test_data, "validation": validation_data}

    dataset = format_data(dataset)

    # Load upload dataset to HF
    push_dataset_to_hf(dataset, hf_config)