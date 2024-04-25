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
    attributes["school_id"] = []
    attributes["school_name"] = []
    attributes["head_name"] = []
    attributes["secretary_name"] = []
    attributes["student_index"] = []
    attributes["student_name"] = []
    attributes["student_gender"] = []
    attributes["student_name_origin"] = []
    attributes["student_courses"] = []
    attributes["layout"] = []
    attributes["academic_years_in_sample"] = []
    attributes["num_subjects"] = []
    attributes["average_grade"] = []
    attributes["blender_mod"] = []
    attributes["rendering_style"] = []
    attributes["shadow_casting"] = []
    attributes["printer_stains"] = []
    attributes["background_elements"] = []
    attributes["modify_mesh"] = []
    attributes["background_material"] = []
    attributes["paper_texture"] = []
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
            try:
                students = school["students"]
                pages_per_student = len(school["template_layout"])
                pages = students * pages_per_student
                num_samples += pages
            except KeyError:
                pass

    return num_samples


def compute_num_students(reqs: dict) -> int:
    num_students = 0
    for language in reqs["samples"].values():
        for school in language.values():
            try:
                if school["include"]:
                    students = school["students"]
                    num_students += students
            except KeyError:
                pass

    return num_students


def compute_num_students_for_lang(reqs: dict, lang: str) -> int:
    num_students = 0

    for school in reqs["samples"][lang.lower()].values():
        try:
            if school["include"]:
                num_students += school["students"]
        except KeyError:
            pass

    return num_students


def compute_render_styles_distribution(mods_distribution: list, b_props: dict) -> list:

    # TODO
    # rendering_styles = b_props["blender"]

    render_styles = [
        (
            random.choices(
                ["studio", "natural", "warm", "scanner"], weights=[0.25, 0.25, 0.2, 0.3]
            )[0]
            if mod == True
            else "N/A"
        )
        for mod in mods_distribution
    ]

    return render_styles


def compute_shadow_casting_distribution(rendering_styles: list) -> list:
    shadows_distribution = []

    for style in rendering_styles:

        if style == "N/A":
            shadows_distribution.append("N/A")
        elif style == "scanner":
            shadows_distribution.append(False)
        else:
            shadows_distribution.append(random.choice([True, False]))

    return shadows_distribution


def compute_printer_stains_distribution(mods_distribution: list) -> list:
    printer_stains = [
        (random.choices([True, False], weights=[0.7, 0.3])[0] if mod == True else "N/A")
        for mod in mods_distribution
    ]

    return printer_stains


def compute_background_elements_distribution(
    rendering_styles: list, b_props: dict
) -> list:
    background_elements = []

    objects = b_props["blender"]["background_objects"]
    choice_options = [False]
    choice_options.extend(objects)
    choice_weights = [1 / len(choice_options) for _ in choice_options]

    for style in rendering_styles:

        if style == "N/A":
            background_elements.append("N/A")
        elif style == "scanner":
            background_elements.append(False)
        else:
            background_elements.append(
                random.choices(choice_options, choice_weights)[0]
            )

    return background_elements


def compute_mesh_modification_distribution(background_elements: list) -> list:
    mesh_mod = []

    for back_objects in background_elements:
        if back_objects == "N/A":
            mesh_mod.append("N/A")
        elif back_objects == False:
            mesh_mod.append(False)
        else:
            mesh_mod.append(True)

    return mesh_mod


def compute_background_material_distribution(rendering_styles, b_props: dict) -> list:

    background_materials = []

    # TODO
    materials = []

    for style in rendering_styles:

        if style == "N/A":
            background_materials.append("N/A")
        elif style == "scanner":
            background_materials.append("white_plastic")
        else:
            background_materials.append(
                random.choice(
                    [
                        "blue_tiles",
                        "cream_terrazo",
                        "multicolor_terrazo",
                        "quartzite",
                        "rusted_metal",
                        "white_oak",
                        "wood_planks",
                    ]
                )
            )

    return background_materials


def compute_paper_texture_distribution(rendering_styles) -> list:
    papers_texture = []

    for style in rendering_styles:
        if style == "N/A":
            papers_texture.append("N/A")
        elif style == "scanner":
            papers_texture.append("scanned")
        else:
            papers_texture.append(
                random.choice(
                    [
                        "blank",
                        "blank_grain",
                        "blank_recycled",
                        "coffee_stain",
                        "folded",
                        "folded_recycled",
                        "food_stain",
                        "marker_stain",
                        "teared",
                        "wrinkled",
                        "wrinkled_recycled",
                        "written_back",
                    ]
                )
            )

    return papers_texture


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
            try:
                if school_values["include"]:
                    selected_languages.append(lang.capitalize())
            except KeyError:
                pass

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
            try:
                if school["include"]:
                    for _ in range(school["students"]):
                        langs.append(language_name)
            except KeyError:
                pass

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
    Get the list of academic years per page from the layout info, the number of
    subjects per academic year, and the layout style for every page.

    This method extracts the list of academic years for each page in the provided layout
    dictionary and checks if the number of academic years matches the number of subjects
    for each page.

    Args:
        layout (dict): A dictionary representing the layout of academic records.

    Returns:
        list: A list of number of subjects per academic year.
        list: A list of academic years per page.
        list: A list of layout style per page.
    """
    table_per_page = [page_value["academic_years"] for page_value in layout.values()]
    subjects_per_table = [page_value["num_subjects"] for page_value in layout.values()]
    layout_style_per_page = [page_value["layout"] for page_value in layout.values()]

    for page_value in layout.values():
        if len(page_value["academic_years"]) != len(page_value["num_subjects"]):
            print("WARNING: every record TABLE should have ONE ACADEMIC YEAR:")
            print("Check your requirements.json file")

    return table_per_page, subjects_per_table, layout_style_per_page


# TODO Make it more pythonic
def fill_blueprint(attributes: dict, reqs: dict, props: dict, b_props: dict) -> dict:
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
    rendering_styles = compute_render_styles_distribution(blender_mod, b_props)
    back_materials = compute_background_material_distribution(rendering_styles, b_props)
    papers_texture = compute_paper_texture_distribution(rendering_styles)
    shadow_castings = compute_shadow_casting_distribution(rendering_styles)
    printer_stains = compute_printer_stains_distribution(blender_mod)
    back_elements = compute_background_elements_distribution(rendering_styles, b_props)
    mesh_modifications = compute_mesh_modification_distribution(back_elements)
    gender = compute_gender_distribution(reqs)
    name_origins = compute_origins_distributions(reqs)
    grades_dists = compute_grades_distributions(reqs, gender, name_origins)

    general_student_index = 0
    for language, language_content in reqs["samples"].items():
        for school_key, school in language_content.items():
            try:
                if school["include"]:
                    students = school["students"]
                    pages_per_student = len(school["template_layout"])
                    pages = students * pages_per_student

                    student_index = 0
                    student_page_index = 0
                    table_per_page, subjects_per_table, layout_style = get_tables_info(
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
                            elif key == "school_id":
                                attributes[key].append(school_key)
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
                                attributes[key].append(
                                    name_origins[general_student_index]
                                )
                            elif key == "layout":
                                attributes[key].append(layout_style[student_page_index])
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
                                attributes[key].append(
                                    grades_dists[general_student_index]
                                )
                            elif key == "replication_done":
                                attributes[key].append(False)
                            elif key == "blender_mod":
                                attributes[key].append(blender_mod[index])
                            elif key == "rendering_style":
                                attributes[key].append(rendering_styles[index])
                            elif key == "background_material":
                                attributes[key].append(back_materials[index])
                            elif key == "paper_texture":
                                attributes[key].append(papers_texture[index])
                            elif key == "shadow_casting":
                                attributes[key].append(shadow_castings[index])
                            elif key == "printer_stains":
                                attributes[key].append(printer_stains[index])
                            elif key == "background_elements":
                                attributes[key].append(back_elements[index])
                            elif key == "modify_mesh":
                                attributes[key].append(mesh_modifications[index])
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
            except KeyError:
                pass

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


def generate_blueprint(reqs: dict, props: dict, b_props: dict) -> None:
    """
    Generates a CSV file containing data with information from user requirements and
    dataset properties.

    This method orchestrates the process of setting up a database structure, populating
    it with data extracted from the 'requirements.json' and 'properties.json' files, and
    then writing this data to a CSV file. It does this by calling 'set_database_headers'
    to initialize the database, 'fill_database' to populate it, and 'write_csv' to write
    the database to a CSV file.

    Args:
        reqs (dict):    User requirements and nested info about samples distribution
                        (languages, schools, or students).
        props (dict):   Properties.
        b_props (dict): Blender properties.
    """

    attributes = set_database_headers()
    attributes = fill_blueprint(attributes, reqs, props, b_props)
    write_csv(attributes)


if __name__ == "__main__":
    root = os.path.join(
        Path(__file__).resolve().parents[1], "replication_pipeline", "assets"
    )
    reqs_path = os.path.join(root, "requirements.json")
    props_path = os.path.join(root, "properties.json")
    blender_props_path = os.path.join(
        Path(__file__).resolve().parents[1], "blender_mod", "assets", "properties.json"
    )
    reqs = read_json(name=reqs_path)
    props = read_json(name=props_path)
    blender_props = read_json(name=blender_props_path)

    generate_blueprint(reqs, props, blender_props)
