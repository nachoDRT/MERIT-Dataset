import json
import base64
from typing import Dict
from io import BytesIO
from os.path import join, abspath, dirname
import os
from datasets import load_dataset, Image
from PIL import Image
import argparse


def encode_image(img):

    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


def load_secrets(file_path: str) -> Dict:
    return load_json(file_path)


def load_config(file_path: str) -> Dict:
    return load_json(file_path)


def load_json(file_path: str) -> Dict:

    with open(file_path, encoding="utf-8") as config_file:
        file_content = json.load(config_file)

    return file_content


def init_apis():

    secrets_path = join(dirname(dirname(abspath(__file__))), "config", "secrets.json")
    secrets = load_secrets(secrets_path)
    os.environ["OPENAI_API_KEY"] = secrets["openai"]
    os.environ["MISTRAL_API_KEY"] = secrets["mistral"]


def detect_json(response: str) -> str:

    start = response.find("{")
    response = response[start:]
    end = response.rfind("}")
    response = response[: end + 1]

    return response


def clean_json(grades: str) -> Dict:

    raw_json = grades.encode("utf-8").decode("unicode_escape")
    corrected_json = raw_json.encode("latin-1").decode("utf-8")

    try:
        grades_dict = json.loads(corrected_json)
    except:
        print(corrected_json)
        grades_dict = {}

    return grades_dict


def get_sample_data(sample):

    img = sample["image"]
    gt = sample["ground_truth"]

    return img, gt


def get_local_sample_data(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    image_path = json_path.with_suffix(".png")
    if not image_path.exists():
        image_path = json_path.with_suffix(".jpg")
    if not image_path.exists():
        image_path = json_path.with_suffix(".jpeg")

    if not image_path.exists():
        print(f"⚠️ Unable to find image {json_path.name}")
        return None, gt

    image = Image.open(image_path)

    return image, gt


def get_dataset_iterator(decode=None):

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit-secret", "fomento", split="test", streaming=True)

    if decode:
        dataset = dataset.cast_column("image", Image(decode=False))

    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_model():

    config_path = join(dirname(dirname(abspath(__file__))), "config", "config.json")
    config = load_config(config_path)

    return config["model"]


def list_fine_tunes(client):
    try:
        response = client.fine_tuning.jobs.list()

        jobs = list(response)
        if not jobs:
            print("No fine-tuning jobs found.")
            return

        print("Fine-tuning jobs found:")
        for job in jobs:
            model_name = getattr(job, "fine_tuned_model", "N/A")
            job_status = job.status
            print(f"Job ID: {job.id}, Model: {model_name}, Status: {job_status}")

    except Exception as e:
        print(f"Error listing fine-tuning jobs: {e}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    elif value.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false, 1/0, yes/no, y/n).")
