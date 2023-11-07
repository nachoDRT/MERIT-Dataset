import json
import os
from pathlib import Path
import pandas as pd
import random
from os.path import dirname, abspath


def read_json(name: str) -> dict:
    """
    Read a JSON file from the current working directory and return its content.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """

    file_path = os.path.join(dirname(abspath(__file__)), name)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def set_database_headers() -> dict:
    """
    Initializes a database dictionary with predefined keys and empty lists as values.

    This method sets up the initial structure of a database dictionary where each key
    represents a specific attribute/column, and each value is an empty list that will
    hold the data entries for that attribute.

    Returns:
        dict: Initialized database dictionary with empty lists as values for each prede-
              fined attribute.
    """

    attributes = {}

    attributes["file_name"] = []
    attributes["language"] = []
    attributes["school_name"] = []
    attributes["head_name"] = []
    attributes["secretary_name"] = []
    attributes["student_id"] = []
    attributes["student_name"] = []
    attributes["student_gender"] = []
    attributes["student_ethnicity"] = []
    attributes["academic_years"] = []
    attributes["average_grades"] = []
    attributes["blender_mod"] = []
    attributes["replication_done"] = []
    attributes["modification_done"] = []

    return attributes


def compute_num_samples(reqs: dict) -> int:
    """
    Calculate the total number of samples to be baked based on the information provided
    in 'requirements.json'.

    This method iterates through a nested structure in the 'reqs' dictionary, summing up
    the product of the number of students and the number of pages per student for each
    school within each language. This total represents the overall number of samples in
    the dataset.

    Args:
        reqs (dict):  User requirements and nested info about samples distribution (lan-
                      guages, schools, or students).
    """

    num_samples = 0
    for language in reqs["samples"].values():
        for school in language.values():
            students = school["students"]
            pages_per_student = school["template_layout"]["pages"]
            pages = students * pages_per_student
            num_samples += pages

    return num_samples


def compute_mods_distribution(reqs: dict) -> list:
    """
    Compute a list of bools based on a given modification factor belonging to [0,1].

    This method generates a list of boolean values representing whether a particular
    sample should be modified in Blender or not. The distribution is determined by a
    modification factor from the "requirements.json" file, and the total number of sam-
    ples as computed by the 'compute_num_samples' method.

    Args:
        reqs (dict):  User requirements and nested info about samples distribution (lan-
                      guages, schools, or students).

    Returns:
        list: A list of boolean values, where each value indicates whether a sample
            should be modified (True) or not (False).
    """

    mod_factor = reqs["blender_modified"]
    mods_distribution = random.choices(
        [True, False], cum_weights=(mod_factor, 1), k=compute_num_samples(reqs)
    )
    return mods_distribution


# TODO Make it more pythonic
def fill_blueprint(attributes: dict, reqs: dict, props: dict) -> dict:
    """
    Populate a database with information from user requirements and dataset properties.
    The output dictionary contains the blueprint that the generator pipeline (replicator
    and, if applicable, modificator) will follow. This method collects the names of the
    samples and includes any other relevant attributes (language, school name, etc.).

    Args:
        database (dict): Database headers without any other data.
        reqs (dict):  User requirements and nested info about samples distribution (lan-
                      guages, schools, or students).
        props (dict): Properties.

    Returns:
        dict: The populated database dictionary.
    """

    # TODO
    """Extract "template_layout" values (json) without the help of the end-user
    (by just reading the WORD)"""

    index = 0
    attributes["blender_mod"] = compute_mods_distribution(reqs)

    general_student_id = 0
    for language, language_content in reqs["samples"].items():
        for school in language_content.values():
            students = school["students"]
            pages_per_student = school["template_layout"]["pages"]
            pages = students * pages_per_student

            student_index = 0
            student_page_index = 0

            for page in range(pages):
                for key in attributes:
                    if key == "file_name":
                        file_name = "".join(
                            [
                                language,
                                "_",
                                school["nickname"],
                                "_",
                                str(student_index).zfill(props["doc_name_zeros_fill"]),
                                "_",
                                str(student_page_index),
                            ]
                        )
                        attributes[key].append(file_name)
                    elif key == "language":
                        attributes[key].append(language)
                    elif key == "school_name":
                        attributes[key].append(school["nickname"])
                    elif key == "head_name":
                        attributes[key].append(str(None))
                    elif key == "secretary_name":
                        attributes[key].append(str(None))
                    elif key == "student_id":
                        attributes[key].append(str(general_student_id))
                    elif key == "student_name":
                        attributes[key].append(str(None))
                    elif key == "student_gender":
                        attributes[key].append("N/A")
                    elif key == "student_ethnicity":
                        attributes[key].append("N/A")
                    elif key == "academic_years":
                        attributes[key].append(str(None))
                    elif key == "average_grades":
                        attributes[key].append(str(None))
                    elif key == "replication_done":
                        attributes[key].append(False)
                    elif key == "modification_done":
                        if attributes["blender_mod"][index] == True:
                            attributes[key].append(False)
                        else:
                            attributes[key].append("N/A")
                    else:
                        pass

                index += 1
                student_page_index += 1

                if (page + 1) % pages_per_student == 0:
                    student_index += 1
                    general_student_id += 1
                    student_page_index = 0

    return attributes


def write_csv(attributes: dict) -> None:
    """
    Write the contents of the database (attributes) dictionary to a CSV file.

    Args:
        attributes (dict): The dictionary whose contents are to be written to a CSV file.
    """

    df = pd.DataFrame(attributes)
    csv_path = os.path.join(Path(__file__).resolve().parent, "dataset_blueprint.csv")
    df.to_csv(csv_path, index=False)


def generate_blueprint(reqs: dict, props: dict) -> None:
    """
    Generates a CSV file containing data with information from user requirements and
    dataset properties.

    This method orchestrates the process of setting up a database structure, populating
    it with data extracted from the 'requirements.json' and 'properties.json' files, and
    then writing this data to a CSV file. It does this by calling 'set_database_headers'
    to initialize the database, 'fill_database' to populate it, and 'write_csv' to write
    the database to a CSV file.

    Args:
        reqs (dict):  User requirements and nested info about samples distribution (lan-
                      guages, schools, or students).
        props (dict): Properties.
    """

    attributes = set_database_headers()
    attributes = fill_blueprint(attributes, reqs, props)
    write_csv(attributes)


if __name__ == "__main__":
    root = os.path.join(
        Path(__file__).resolve().parents[1], "replication_pipeline", "assets"
    )
    reqs_path = os.path.join(root, "requirements_v2.json")
    props_path = os.path.join(root, "properties.json")
    reqs = read_json(name=reqs_path)
    props = read_json(name=props_path)

    generate_blueprint(reqs, props)
