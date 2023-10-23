import os
import glob
import json


DOCS_Z_FILL = 4


def delete_files(reqs: dict, props: dict, ds_path: str):
    """
    Delete generated files for specific student records. This method iterates through
    student numbers starting from a specified value, deleting all files related to each
    student number within a designated directory. The iteration stops when no more files
    are found for a particular student number.

    Args:
        reqs (dict): Requirements (includes the student number from where to initiate
                     deletion).
        props (dict): Properties used in file naming conventions.
        ds_path (str): The path of the directory where the files are stored.
    """

    i = reqs["first_student"]

    while True:
        glob_expr = ds_path + "/*/*{}*".format(
            str(i).zfill(props["doc_name_zeros_fill"])
        )
        files = glob.glob(glob_expr)

        # Delete files
        i = i + 1

        if len(files) == 0:
            break
        print("Deleting student files: {}".format(i))
        for file in files:
            os.remove(file)


def read_json(name: str):
    """
    Read a JSON file from the current working directory and return its content.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """
    file_path = os.path.join(os.getcwd(), name)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data
