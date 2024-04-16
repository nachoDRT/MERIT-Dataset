import wandb
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, List, Tuple, Dict

ROOT = os.path.abspath(__file__)


def get_config() -> Tuple[List, str]:
    file_path = os.path.join(Path(ROOT).parents[0], "config", "config.json")

    with open(file_path, "r") as file:
        data = json.load(file)

    training_keys = data["relevant_training_keys"]
    project_name = "".join([data["entity"], "/", data["project"]])

    return training_keys, project_name


def configure_wandb(project_path: str) -> Iterable:
    api = wandb.Api()
    runs = api.runs(project_path)

    return runs


def extend_results(data_to_extend: List, extensions: Dict):

    for extension in extensions:
        data_to_extend.extend(extension.values())

    return data_to_extend


def get_headers(keys: List) -> List:
    headers = ["training_name"]
    headers.extend(keys)

    return headers


def get_training_summary_from_wandb(runs: Iterable, keys: List) -> pd.DataFrame:
    data = []

    for run in tqdm(runs):
        if run.state == "finished":
            print(f"Gathering resuts from: {run.name}")
            run_data = [run.name]
            training_results = [row for row in run.scan_history(keys=keys)]
            extended_data = extend_results(run_data, training_results)
            data.append(extended_data)

    df = pd.DataFrame(data)
    df.columns = get_headers(keys)

    return df


def save_data(data: pd.DataFrame):
    csv_path = os.path.join(Path(ROOT).parents[0], "training_results.csv")
    data.to_csv(csv_path, index=False)


if __name__ == "__main__":
    training_keys, project_path = get_config()
    training_runs = configure_wandb(project_path)
    training_data = get_training_summary_from_wandb(training_runs, training_keys)
    save_data(training_data)
