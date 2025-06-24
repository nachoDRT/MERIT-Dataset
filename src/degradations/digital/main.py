import argparse
from degradations import *
from utils import *
from push_to_hub import *
import numpy as np
import gc

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
        zoom = 0.25
        if zoom:
            degradation_subset_name = f"{language}-digital-{degradation}-{str(zoom)}-degradation-{data_format}"
        else:
            degradation_subset_name = f"{language}-digital-{degradation}-degradation-{data_format}"
        merit_subset_name = f"{language}-digital-seq"

        for split in splits:
            print(f"Generating {split} {degradation} samples")
            merit_subset_iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)
            split_subset = generate_zoom_samples(merit_subset_iterator, scale=zoom)
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

    elif args.degradation.lower() in ("noisy"):

        merit_subset_name = f"{language}-digital-seq"
        noisy_subset = f"{language}-digital-{degradation}-degradation-{data_format}"

        snr_ratios = []

        for split in splits:
            print(f"Generating {split} {degradation} samples")

            iterator, _ = get_merit_dataset_iterator(merit_subset_name, split)

            # Process batches
            ds_split, split_snr = generate_noisy_samples_stream(
                iterator,
                batch_size=124,
                seed=42,
            )

            # Push the split -> Free RAM
            push_dataset_to_hf({split: ds_split}, noisy_subset)
            # push_dataset_to_hf(ds_split, repo_split)

            snr_ratios.extend(split_snr)

            # delete memory associated to the split
            del ds_split, split_snr
            gc.collect()

        mean_ratio = float(np.mean(snr_ratios))
        mean_db = float(10 * np.log10(mean_ratio))
        std_db = float(np.std(10 * np.log10(snr_ratios)))

        print(f"SNR(dataset) = {mean_db:.2f} dB ± {std_db:.2f} dB")

    else:
        print(f"Degradation called {degradation} has not been implemented yet.")
