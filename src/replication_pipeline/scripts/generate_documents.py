import tqdm
import general_helper as ghelper
import numpy as np
from pathlib import Path
from typing import List, Dict
import os
import pandas as pd
import copy
import ast

import records_creator
import annotations_creator

import cv2

OUTPUT_ANNOTATIONS_PDF = False


def get_reqs_and_props() -> tuple:
    """
    Load and return the contents of 'properties.json' and 'requirements.json' files
    located in the "assets" directory.

    These JSON files are expected to contain configuration values and requirements
    needed to configure the output.

    Returns:
        tuple: A tuple containing two dictionaries:
            - reqs (dict): A dictionary with data from 'requirements.json'.
            - props (dict): A dictionary with data from 'properties.json'.
    """

    properties_path = os.path.join(
        Path(__file__).resolve().parents[1], "assets", "properties.json"
    )
    requirements_path = os.path.join(
        Path(__file__).resolve().parents[1], "assets", "requirements.json"
    )
    props = ghelper.read_json(name=properties_path)
    reqs = ghelper.read_json(name=requirements_path)

    return reqs, props


def define_paths(language: str, school: str) -> dict:
    """
    Construct a dictionary containing file paths used in the script, based on a given
    school nickname. The paths are constructed relative to the parent directory of the
    current working directory.

    Args:
        language (str): The language used: English, Spanish, etc.
        school (str): The nickname of the school, used to customize the base path for
        this particular school's data.

    Returns:
        dict: A dictionary where keys are descriptive names of paths, and values are the
        corresponding file paths.
    """

    paths = {}

    # paths["root_assets"] = os.path.join(
    #     os.path.join(Path(__file__).resolve().parents[1]), language
    # )
    paths["root_output"] = os.path.join(
        Path(__file__).resolve().parents[3], "data", "original"
    )
    paths["base_path"] = os.path.join(paths["root_output"], language, school)
    paths["res_path"] = os.path.join(
        os.path.join(Path(__file__).resolve().parents[1]), "assets", language
    )

    paths["template_path"] = os.path.join(
        Path(__file__).resolve().parents[1],
        "templates",
        language,
        school,
        "template.docx",
    )

    paths["ds_path"] = os.path.join(
        Path(__file__).resolve().parents[3],
        "data",
        "original",
        language,
        school,
        "dataset_output",
    )
    paths["annotations_path"] = os.path.join(paths["ds_path"], "annotations")
    paths["images_path"] = os.path.join(paths["ds_path"], "images")
    paths["pdf_path"] = os.path.join(paths["ds_path"], "synthetic_pdf_docs")

    return paths


def get_students_num_subjects(indices: List, df: pd.DataFrame):
    """
    Extracts and processes the number of subjects for every student that has replicas
    still to be done.

    This method iterates through the list of indices that determine a student change in
    the blueprint. The method uses each pair of adjacent indices to slice a portion of
    the DataFrame 'df' (blueprint). It extracts the 'num_subjects' column from these
    slices, processes the values (the number os subjects), and aggregates them into a
    list.

    Parameters:
        indices (List): A list of indices used to slice the DataFrame. Each pair of ad-
                        jacent indices defines the start and end (exclusive) of a slice.
                        These indices indicate a new student in the blueprint.
        df (pd.DataFrame): The blueprint.

    Returns:
        List: Each element in the list corresponds to one student. Every element in the
        list is another list with the number of subjects for that student's sample.

    Note:
        The 'num_subjects' column in the DataFrame is expected to contain string repre-
        sentations of lists. This function converts these strings into actual lists and
        extracts the first element of each list.
    """

    selected_items = []
    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1]
        items = df.loc[start_index : end_index - 1, "num_subjects"].tolist()
        items = [ast.literal_eval(item)[0] for item in items]
        selected_items.append(items)

    return selected_items


def check_chaotic_crash(
    n_subjects: List,
    courses: List[str],
    index: int,
    df: pd.DataFrame,
) -> Dict:
    """
    Check if a 'chaotic crash' happened: a pipeline crash happening in the middle of the
    documents generation of one student.

    This method compares the lengths of 'n_subjects' and 'courses' lists. If they are
    not equal, it indicates a 'chaotic crash', and various details related to the crash
    are gathered from the blueprint and returned in a dictionary.

    Parameters:
        n_subjects (List): A list containing the number of subjects for one student.
        courses (List[str]): A list of course names.
        index (int): The index in the DataFrame 'df' (blueprint) to extract crash de-
        tails from.
        df (pd.DataFrame): A pandas DataFrame containing the blueprint.

    Returns:
        Dict: A dictionary with details about the 'chaotic crash'. If a crash has happe-
              ned, it includes keys like 'courses', 'n_subjects', 'language', 'direc-
              tor', 'secretary', 'student_name', and 'school_name'. If no crash happe-
              ned, it only contains the key 'happened' set to False.
    """

    crash = {}
    crash["happened"] = False

    if len(n_subjects) != len(courses):
        crash["happened"] = True
        crash["courses"] = courses[-(len(n_subjects)) :]
        crash["n_subjects"] = n_subjects
        crash["language"] = df.at[index - 1, "language"]
        crash["director"] = df.at[index - 1, "head_name"]
        crash["secretary"] = df.at[index - 1, "secretary_name"]
        crash["student_name"] = df.at[index - 1, "student_name"]
        crash["school_name"] = df.at[index - 1, "school_name"]
        crash["student_sample_index"] = -len(n_subjects)

    return crash


def get_courses_indices_in_sample(df: pd.DataFrame, index: int, courses: list):
    """
    Extract the indices of the academic yars present in one sample. The indeces are re-
    lative to one unique student.

    Parameters:
        df (pd.DataFrame): The Blueprint data content
        index (int): The row index in the DataFrame i.e. the sample index.
        courses (list): The list of course names or academic years relevant for specific
                        student.

    Returns:
        list: A list of indices corresponding to those academic years present in one
              specific sample
    """

    academic_years_in_sample = df.at[index, "academic_years_in_sample"]
    student_sample_indeces = [
        i
        for i, academic_year in enumerate(courses)
        if academic_year in academic_years_in_sample
    ]

    return student_sample_indeces


def compute_average_grade(curriculum: list):
    """
    Compute the average grade from a list of subjects, each represented as a dictionary.

    This method iterates through each subject in the 'curriculum' list. Each subject is
    expected to be a dictionary where one of the values is another dictionary containing
    the key 'grade'. It extracts all these 'grade' values, calculates their average, and
    returns it.

    Parameters:
        curriculum (list): A list where each element is a dictionary representing a sub-
                           ject. Each subject's dictionary should contain a key-value
                           pair where the value is another dictionary with the key
                           'grade'.

    Returns:
        float: The average of all the 'grade' values found in the curriculum.
    """

    grades = []

    for subject in curriculum:
        for subject_values in subject.values():
            grades.append(subject_values["grade"])

    grades = np.array(grades)
    average = grades.mean()

    return average


def compute_average_grades(indices: list, record: records_creator.SchoolRecord):
    """
    Computes the average grades for the given indices in a student's curriculum record.
    The indices point towards the academic years present in one specific sample. Then
    the method computes the average.

    Parameters:
        indices (list): A list of integers representing the indices of the academic
                        years present in one specific sample.

        record (records_creator.SchoolRecord): A student's record.

    Returns:
        list: A list containing one average (float) for every academic year present in
              the sample.
    """

    averages = []
    for index in indices:
        averages.append(
            round(
                compute_average_grade(record.student.curriculum[index]),
                2,
            )
        )

    return averages


def load_assets(
    school_name: str,
    res_path: str,
    assets_to_load: List = ["signatures", "signature_maps", "stamps", "stamp_maps"],
):
    assets = {}
    for asset_to_load in assets_to_load:
        file_name = "".join([asset_to_load[:-1], "_", school_name, ".png"])
        asset_path = os.path.join(res_path, asset_to_load, file_name)
        assets[asset_to_load[:-1]] = load_asset_img(asset_path)

    return assets


def load_asset_img(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def create_documents(df: pd.DataFrame, blueprint_path: str):
    """
    Generates synthetic student records and related documents (PDF and PNG), along with
    their annotations. This method iterates through the students still lacking replicas,
    creating synthetic records, documents, and annotations for each. It also manages the
    tagging of subjects, grades, and courses within the annotations, and writes the
    annotations to a JSON file. Optionally, it can output a PDF view of the annotations.

    Args:
        df (pd.DataFrame): The blueprint content.
        blueprint_path (str): The blueprint path.
    """

    # TODO monolitic method, subdivide in speficic methods

    # Select only those samples still to be generated
    mask = df["replication_done"] == False
    filtered_df = df[mask]

    language_changes = filtered_df["language"] != filtered_df["language"].shift(1)
    school_changes = filtered_df["school_name"] != filtered_df["school_name"].shift(1)
    student_changes = filtered_df["student_index"] != filtered_df[
        "student_index"
    ].shift(1)

    # Filter indeces where there are changes in language and school names
    language_change_indices = filtered_df.index[language_changes]
    school_change_indices = filtered_df.index[school_changes]
    student_changes_indices = filtered_df.index[student_changes]
    last_student_index = df.shape[0]
    student_changes_indices = student_changes_indices.tolist()
    student_changes_indices.append(last_student_index)

    students_n_subjects = get_students_num_subjects(
        student_changes_indices, filtered_df
    )

    retrieved_info = {}
    index_offset = int(df.at[df["replication_done"].sum(), "student_index"])

    for index, sample_name in tqdm.tqdm(
        filtered_df["file_name"].items(), total=filtered_df.shape[0]
    ):
        # TODO check crash only in the first iteration of the loop
        new_school = False
        new_student = False

        courses = ast.literal_eval(df.at[index, "student_courses"])

        student_index = int(df.at[index, "student_index"])
        student_index -= index_offset
        n_subjects = students_n_subjects[student_index]

        crash_data = check_chaotic_crash(n_subjects, courses, index, df)

        if crash_data["happened"]:
            n_subjects = crash_data["n_subjects"]
            courses = crash_data["courses"]
            retrieved_info["director"] = crash_data["director"]
            retrieved_info["secretary"] = crash_data["secretary"]
            retrieved_info["student_name"] = crash_data["student_name"]
            retrieved_info["re_spawn_student"] = True
            school = crash_data["school_name"]
            lang = crash_data["language"]
            # student_sample_index = crash_data["student_sample_index"]
            paths = define_paths(language=lang, school=school)

        else:
            lang = df.at[index, "language"]

            if index in language_change_indices.tolist():
                school = df.at[index, "school_name"]
                paths = define_paths(language=lang, school=school)
                new_school = True

            elif index in school_change_indices.tolist():
                school = df.at[index, "school_name"]
                paths = define_paths(language=lang, school=school)
                new_school = True

            if index in student_changes_indices:
                new_student = True

        # Create synthetic student records
        if new_student:
            record = records_creator.SchoolRecord(
                sample_name,
                paths,
                props,
                reqs,
                lang,
                courses,
                n_subjects,
                retrieved_info=retrieved_info,
                new_school=new_school,
                new_student=new_student,
            )

            png_assets = load_assets(school, paths["res_path"])
            pdf_file_path = record.create_pdf()

        if crash_data["happened"]:
            # TODO make sure you recover every student piece of data
            record.student._name = crash_data["student_name"].split()[0]
            record.student._first_surname = crash_data["student_name"].split()[1]
            record.student._second_surname = crash_data["student_name"].split()[2]

        # Create pdf and png documents
        png_paths = record.create_pngs(png_assets)

        # Create annotations
        annotations_class = annotations_creator.AnnotationsCreator(
            pdf_file_path, paths, props=props, reqs=reqs
        )
        annotations_class.create_annotations(pdf_file_path)

        # For this sample, get the indices of the academic years present
        student_sample_indices = get_courses_indices_in_sample(df, index, courses)

        averages = compute_average_grades(student_sample_indices, record)

        # Tag subjects and grades
        annotations_class.new_update_subject_grades_tags(
            record.student.curriculum, n_subjects
        )

        # Tag courses
        annotations_class.update_annotations_tags_courses()
        annotations_class.sort_annotations()

        # Write to json
        annotations_class.dump_annotations_json(png_paths)

        # Update blueprint
        retrieved_info = record.get_info_retrieval()
        df.at[index, "head_name"] = retrieved_info["director"]
        df.at[index, "secretary_name"] = retrieved_info["secretary"]
        df.at[index, "student_name"] = retrieved_info["student_name"]
        df.at[index, "replication_done"] = True
        df.at[index, "average_grade"] = averages
        df.to_csv(blueprint_path, index=False)


def get_blueprint():
    """
    Retrieves the dataset blueprint from a CSV file.

    This method builds the path to the CSV blueprint ('dataset_blueprint.csv'),
    located in the 'dashboard' directory. It then reads the CSV file into a pandas
    DataFrame.

    Returns:
        tuple: A tuple containing two elements:
            - pandas.DataFrame: The DataFrame created from the CSV file.
            - str: The file path to the CSV file.
    """

    blueprint_path = os.path.join(
        Path(__file__).resolve().parents[2], "dashboard", "dataset_blueprint.csv"
    )
    blueprint_df = pd.read_csv(blueprint_path)

    return blueprint_df, blueprint_path


if __name__ == "__main__":
    blueprint_df, blueprint_path = get_blueprint()
    blueprint_copy_df = copy.deepcopy(blueprint_df)

    reqs, props = get_reqs_and_props()
    create_documents(blueprint_df, blueprint_path)
    # Recover the original csv: just useful for debugging purposes
    # blueprint_copy_df.to_csv(blueprint_path, index=False)
