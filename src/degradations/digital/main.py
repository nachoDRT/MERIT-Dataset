import argparse
from degradations import *
from utils import *
from push_to_hub import *

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", required=True, type=str)
    parser.add_argument("--language", required=True, type=str)
    parser.add_argument("--format", required=True, type=str)
    args = parser.parse_args()

    degradation = args.degradation
    language = args.language
    data_format = args.format

    merit_subset_name = f"{language}-digital-token-class"
    splits = get_merit_dataset_splits(merit_subset_name)

    dataset = []

    if args.degradation.lower() in ("paragraph"):

        degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_paragraph_samples(merit_subset_iterator, language)
            dataset.append((split, split_subset))
        dataset = format_data(dict(dataset))
        push_dataset_to_hf(dataset, degradation_subset_name)

    elif args.degradation.lower() in ("line"):

        degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_line_samples(merit_subset_iterator, language)
            dataset.append((split, split_subset))
        dataset = format_data(dict(dataset))
        push_dataset_to_hf(dataset, degradation_subset_name, repo_name="merit")

    elif args.degradation.lower() in ("rotation"):
        degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"
        merit_subset_name = f"{language}-digital-seq"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_rotation_samples(merit_subset_iterator)
            dataset.append((split, split_subset))
        dataset = format_data(dict(dataset))
        push_dataset_to_hf(dataset, degradation_subset_name)

    elif args.degradation.lower() in ("zoom"):
        degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"
        merit_subset_name = f"{language}-digital-seq"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_zoom_samples(merit_subset_iterator)
            dataset.append((split, split_subset))
        dataset = format_data(dict(dataset))
        push_dataset_to_hf(dataset, degradation_subset_name)

    elif args.degradation.lower() in ("rotation-zoom"):
        degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"
        merit_subset_name = f"{language}-digital-seq"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_rotation_zoom_samples(merit_subset_iterator)
            dataset.append((split, split_subset))
        dataset = format_data(dict(dataset))
        push_dataset_to_hf(dataset, degradation_subset_name)

    else:
        print(f"Degradation called {degradation} has not been implemented yet.")
