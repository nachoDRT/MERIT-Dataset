import os
import json
from os.path import abspath, dirname, join
from pathlib import Path


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def get_data(school: str):
    imgs = []
    annotations = []

    directory = join(dirname(dirname(abspath(__file__))), "data", school)

    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    for json_file in Path(directory).glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                annotations.append(data)

            base_name = json_file.stem
            image_path = None

            for ext in ["png", "jpg", "jpeg"]:
                possible_image = Path(directory) / f"{base_name}.{ext}"
                if possible_image.exists():
                    image_path = possible_image
                    break

            if image_path:
                with open(image_path, "rb") as img_file:
                    imgs.append(img_file.read())
            else:
                print(f"Warning: Img {json_file.name} not found")

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    return {"image": imgs, "ground_truth": annotations}
