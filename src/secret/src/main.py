import argparse
from utils import *
from push_to_hub import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--school", required=True, type=str)
    args = parser.parse_args()
    school = args.school

    split_subset = get_data(school)

    dataset = [("test", split_subset)]
    dataset = format_data_secret(dict(dataset))
    push_dataset_to_hf(dataset, school, repo_name="merit-secret")
