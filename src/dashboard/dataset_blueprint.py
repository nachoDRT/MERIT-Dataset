import json
import os
from pathlib import Path
import pandas as pd
import random
from os.path import dirname, abspath
import numpy as np


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
    attributes["student_index"] = []
    attributes["student_name"] = []
    attributes["student_gender"] = []
    attributes["student_name_origin"] = []
    attributes["student_courses"] = []
    attributes["academic_years_in_sample"] = []
    attributes["num_subjects"] = []
    attributes["average_grade"] = []
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
            pages_per_student = len(school["template_layout"])
            pages = students * pages_per_student
            num_samples += pages

    return num_samples


def compute_num_students(reqs: dict) -> int:
    num_students = 0
    for language in reqs["samples"].values():
        for school in language.values():
            if school["include"]:
                students = school["students"]
                num_students += students

    return num_students


def compute_num_students_for_lang(reqs: dict, lang: str) -> int:
    num_students = 0

    for school in reqs["samples"][lang.lower()].values():
        if school["include"]:
            num_students += school["students"]

    return num_students


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


def compute_gender_distribution(reqs: dict) -> list:
    f_w = reqs["female_proportion"]
    m_w = reqs["male_proportion"]

    gender_distribution = random.choices(
        ["female", "male"], cum_weights=(f_w, f_w + m_w), k=compute_num_students(reqs)
    )

    return gender_distribution


def compute_origins_distributions(reqs: dict) -> list:
    selected_languages = []

    for lang, lang_values in reqs["samples"].items():
        for school_values in lang_values.values():
            if school_values["include"]:
                selected_languages.append(lang.capitalize())

    total_origins_distribution = []
    for lang, origins in reqs["origins"].items():
        weights = [origin["proportion"] / 100 for origin in origins.values()]
        available_origins = [origin.lower() for origin in origins]
        # weights = [origin / 100 for origin in origins.values()]
        # cum_weights = [weight if i==0 else weight for i, weight in enumerate(weights)]
        cum_weights = []
        for i, weight in enumerate(weights):
            if i == 0:
                cum_weights.append(weight)
                last_weight = weight
            elif i + 1 == len(weights):
                cum_weights.append(1)
            else:
                cum_weights.append(last_weight + weight)
                last_weight = weight

        cum_weights = [round(weight, 1) for weight in cum_weights]
        # The distribtuion for the considered main language
        total_origins_distribution.extend(
            random.choices(
                available_origins,
                cum_weights=cum_weights,
                k=compute_num_students_for_lang(reqs, lang),
            )
        )

    return total_origins_distribution


def build_language_distribution(reqs: dict):
    langs = []
    for language_name, language_info in reqs["samples"].items():
        for school in language_info.values():
            if school["include"]:
                for _ in range(school["students"]):
                    langs.append(language_name)

    return langs


def compute_grades_distributions(reqs: dict, gender: list, origin: list) -> list:
    grades_dist = []
    num_students = compute_num_students(reqs)
    langs = build_language_distribution(reqs)

    for index in range(num_students):
        dist = {}

        lang = langs[index]
        student_gender = gender[index]
        student_origin = origin[index]

        gender_bias_key = "".join([student_gender, "_bias_distribution"])
        mean_g = reqs[gender_bias_key]["average"]
        dev_g = reqs[gender_bias_key]["deviation"]

        mean_o = reqs["origins"][lang][student_origin]["average"]
        dev_o = reqs["origins"][lang][student_origin]["deviation"]

        dist["mean_gender"] = mean_g
        dist["dev_gender"] = dev_g

        dist["mean_origin"] = mean_o
        dist["dev_origin"] = dev_o

        grades_dist.append(dist)

    return grades_dist


def get_tables_info(layout: dict):
    """
    Get the list of academic years per page from the layout info and the number of
    subjects per academic year.

    This method extracts the list of academic years for each page in the provided layout
    dictionary and checks if the number of academic years matches the number of subjects
    for each page.

    Args:
        layout (dict): A dictionary representing the layout of academic records.

    Returns:
        list: A list of academic years per page.
    """
    table_per_page = [page_value["academic_years"] for page_value in layout.values()]
    subjects_per_table = [page_value["num_subjects"] for page_value in layout.values()]
    for page_value in layout.values():
        if len(page_value["academic_years"]) != len(page_value["num_subjects"]):
            print("WARNING: every record TABLE should have ONE ACADEMIC YEAR:")
            print("Check your requirements.json file")

    return table_per_page, subjects_per_table


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
    # General attributes
    # TODO "blender_mod" should be a dashboard input in the future
    blender_mod = compute_mods_distribution(reqs)
    gender = compute_gender_distribution(reqs)
    name_origins = compute_origins_distributions(reqs)
    grades_dists = compute_grades_distributions(reqs, gender, name_origins)

    general_student_index = 0
    for language, language_content in reqs["samples"].items():
        for school in language_content.values():
            if school["include"]:
                students = school["students"]
                pages_per_student = len(school["template_layout"])
                pages = students * pages_per_student

                student_index = 0
                student_page_index = 0
                table_per_page, subjects_per_table = get_tables_info(
                    school["template_layout"]
                )
                student_courses = [
                    course for courses in table_per_page for course in courses
                ]
                for page in range(pages):
                    for key in attributes:
                        if key == "file_name":
                            file_name = "".join(
                                [
                                    language,
                                    "_",
                                    school["nickname"],
                                    "_",
                                    str(student_index).zfill(
                                        props["doc_name_zeros_fill"]
                                    ),
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
                        elif key == "student_index":
                            attributes[key].append(str(general_student_index))
                        elif key == "student_name":
                            attributes[key].append(str(None))
                        elif key == "student_gender":
                            attributes[key].append(gender[general_student_index])
                        elif key == "student_name_origin":
                            attributes[key].append(name_origins[general_student_index])
                        elif key == "academic_years_in_sample":
                            attributes[key].append(
                                str(table_per_page[student_page_index])
                            )
                        elif key == "student_courses":
                            attributes[key].append(student_courses)
                        elif key == "num_subjects":
                            attributes[key].append(
                                str(subjects_per_table[student_page_index])
                            )
                        elif key == "average_grade":
                            attributes[key].append(grades_dists[general_student_index])
                        elif key == "replication_done":
                            attributes[key].append(False)
                        elif key == "blender_mod":
                            attributes[key].append(blender_mod[index])
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
                        general_student_index += 1
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
    reqs_path = os.path.join(root, "requirements.json")
    props_path = os.path.join(root, "properties.json")
    reqs = read_json(name=reqs_path)
    props = read_json(name=props_path)

    generate_blueprint(reqs, props)
