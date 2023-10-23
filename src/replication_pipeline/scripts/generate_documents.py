import tqdm
import general_helper as ghelper
from pathlib import Path
import os

import records_creator
import annotations_creator

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
        Path(os.getcwd()).parent, "assets", "properties.json"
    )
    requirements_path = os.path.join(
        Path(os.getcwd()).parent, "assets", "requirements.json"
    )
    props = ghelper.read_json(name=properties_path)
    reqs = ghelper.read_json(name=requirements_path)

    return reqs, props


def define_paths(school_nickname: str) -> dict:
    """
    Construct a dictionary containing file paths used in the script, based on a given
    school nickname. The paths are constructed relative to the parent directory of the
    current working directory.

    Args:
        school_nickname (str): The nickname of the school, used to customize the base
        path for this particular school's data.

    Returns:
        dict: A dictionary where keys are descriptive names of paths, and values are the
        corresponding file paths.
    """

    paths = {}

    paths["root"] = Path(os.getcwd()).parent

    paths["base_path"] = os.path.join(paths["root"], school_nickname)
    paths["res_path"] = os.path.join(paths["root"], "assets")

    paths["ds_path"] = os.path.join(paths["base_path"], "dataset_output")
    paths["template_path"] = os.path.join(paths["base_path"], "template.docx")

    paths["annotations_path"] = os.path.join(paths["ds_path"], "annotations")
    paths["images_path"] = os.path.join(paths["ds_path"], "images")
    paths["pdf_path"] = os.path.join(paths["ds_path"], "synthetic_pdf_docs")

    return paths


def create_documents(reqs: dict, props: dict, paths: dict) -> None:
    """
    Generates synthetic student records and related documents (PDF and PNG), along with
    their annotations. This method iterates through a range of students, creating syn-
    thetic records, documents, and annotations for each. It also manages the tagging of
    subjects, grades, and courses within the annotations, and writes the annotations to
    a JSON file. Optionally, it can output a PDF view of the annotations.

    Args:
        reqs (dict): Requirements for document generation.
        props (dict): Properties for document generation and annotation.
        paths (dict): File paths used for document generation, annotation, and storage.
    """

    first_student = reqs["first_student"]
    students = reqs["students"]

    print(f"Starting at student: {first_student} of {students}")

    for student_n in tqdm.trange(
        first_student, students, initial=first_student, total=students
    ):
        # Create synthetic student records
        record = records_creator.SchoolRecord(student_n, paths, props, reqs)

        # Create pdf and png documents
        pdf_file_path = record.create_pdf()
        png_paths = record.create_pngs()

        # Create annotations
        annotations_class = annotations_creator.AnnotationsCreator(
            pdf_file_path, paths, props=props, reqs=reqs
        )
        annotations_class.create_annotations(pdf_file_path)

        # Tag subjects and grades
        annotations_class.update_subject_grades_tags(
            record.student.curriculum, props["courses_pages_array"]
        )

        # Tag courses
        annotations_class.update_annotations_tags_courses()
        annotations_class.sort_annotations()

        # Write to json
        annotations_class.dump_annotations_json(png_paths)

        # Optionally output annotations view of pdf
        if OUTPUT_ANNOTATIONS_PDF:
            annotations_view = annotations_creator.AnnotationsView(
                annotations_class.annotations, pdf_file_path, reqs
            )
            annotations_view.output_annotations_view()
            annotations_view.output_annotations_words_view()

        # Next time start from next student
        first_student = student_n


if __name__ == "__main__":
    # Load requirements and properties
    reqs, props = get_reqs_and_props()

    # Define intermediate and output paths
    paths = define_paths(school_nickname=reqs["school_nickname"])

    # Delete previous files starting from "first_student" onwards
    ghelper.delete_files(reqs, props, ds_path=paths["ds_path"])
    create_documents(reqs=reqs, props=props, paths=paths)
