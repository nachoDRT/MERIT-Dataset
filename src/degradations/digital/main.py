import argparse
from degradations import *
from utils import *

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", required=True, type=str)
    parser.add_argument("--language", required=True, type=str)
    args = parser.parse_args()

    degradation = args.degradation
    language = args.language

    merit_subset_name = f"{language}-digital-token-class"
    merit_subset = get_merit_dataset_iterator(merit_subset_name)

    if args.degradation.lower() in ("paragraph"):
        print("Generating paragraph samples")
        generate_paragraph_samples(merit_subset)
    else:
        print(f"Degradation called {degradation} has not been implemented yet.")
