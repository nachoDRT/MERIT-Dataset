import argparse
from degradations import *
from utils import *

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--degradation", required=True, type=str)
    parser.add_argument("--language", required=True, type=str)
    parser.add_argument("--school", required=True, type=str)
    args = parser.parse_args()

    degradation = args.degradation
    language = args.language
    school = args.school

    merit_subset = get_merit_subset_paths(language, school)

    if args.degradation.lower() in ("paragraph"):
        generate_paragraph_samples(merit_subset)
    else:
        print(f"Degradation called {degradation} has not been implemented yet.")
