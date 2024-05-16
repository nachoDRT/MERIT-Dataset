import os
import json
import copy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = os.path.join(Path(__file__).resolve().parents[3], "data", "modified")
BLUEPRINT = os.path.join(
    Path(__file__).resolve().parents[2], "dashboard", "dataset_blueprint.csv"
)


def read_json(name: str):
    """
    Read a JSON file from the current working directory and returns its content.

    Args:
        name (str): The name of the JSON file (including the .json extension).

    Returns:
        dict: A dictionary containing the data from the JSON file.

    """

    file_path = os.path.join(os.getcwd(), name)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data


def write_json(data: dict, file_path: str) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        data (dict): The data to be written to the file.
        file_path (str): The path of the file to write to.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_blueprint():
    """
    Retrieves the dataset blueprint from a CSV file.

    This method builds the path to the CSV blueprint ('dataset_blueprint.csv'),
    located in the 'dashboard' directory. It then reads the CSV file into a pandas
    DataFrame.

    Returns:
        tuple: A tuple containing two elements:
            - pandas.DataFrame: The DataFrame created from the CSV file.
    """

    blueprint_df = pd.read_csv(BLUEPRINT)

    return blueprint_df


def postprocess():
    """Modified samples in Blender have bounding boxes defined by their four corners
    (since they might be rotated). On the other hand, LayoutLM models are limited to
    analyzing bounding boxes defined by just two points. This script iterates over the
    dataset folder structure, extracts the first and third bounding box vertices for
    every word, and gathers all the new annotation data in a new folder called
    'annotations_post'."""

    # Get the blueprint
    blueprint_df = get_blueprint()

    # Loop over the languages
    for language in sorted(os.listdir(ROOT)):
        lang_schools_dir = os.path.join(ROOT, language)

        # Loop over the schools
        for school in sorted(os.listdir(lang_schools_dir)):
            print(
                f"Postprocessing samples in {language.capitalize()} for {school.capitalize()}"
            )
            annotations_dir = os.path.join(
                lang_schools_dir,
                school,
                "dataset_output",
                "annotations",
            )
            saving_data = {"language": language, "school": school}
            blueprint_df = postprocess_annotations(
                annotations_dir, blueprint_df, saving_data
            )
    save_blueprint_data(blueprint_df)


def check_bbox(bbox_vertices: list):
    bbox_flag = False

    x0 = bbox_vertices[0]
    y0 = bbox_vertices[1]
    x1 = bbox_vertices[2]
    y1 = bbox_vertices[3]

    # Bbox out of bounds of the image dims
    if any(x < 0 for x in bbox_vertices):
        print(bbox_vertices, 1)
        bbox_flag = True
    elif y0 > 1920 or y1 > 1920:
        print(bbox_vertices, 2)
        bbox_flag = True
    elif x0 > 1080 or x1 > 1080:
        print(bbox_vertices, 3)
        bbox_flag = True

    # No area bbox
    elif x0 - x1 == 0 or y0 - y1 == 0:
        bbox_flag = True

    return bbox_flag


def inscribed_rect_coords(box_coords: list) -> list:

    x_coords = [box_coords[i] for i in range(0, 8, 2)]
    y_coords = [box_coords[i + 1] for i in range(0, 8, 2)]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    return [min_x, min_y, max_x, max_y]


def postprocess_bboxes(name: str, data: dict, b_df: pd.DataFrame):
    postprocessed_data = copy.deepcopy(data)
    form = postprocessed_data["form"]

    bboxes_out = False

    # Iterate over the info in the annotation file
    for segment in form:
        # Process the segment box
        segment_bbox = segment["box"]

        # processed_segment_bbox = [
        #     segment_bbox[0],
        #     segment_bbox[1],
        #     segment_bbox[4],
        #     segment_bbox[5],
        # ]
        processed_segment_bbox = inscribed_rect_coords(segment_bbox)

        segment["box"] = processed_segment_bbox
        segment_flag = check_bbox(segment["box"])

        # Process every word in the segment
        words_flag = False
        for word in segment["words"]:
            word_bbox = word["box"]
            # processed_word_bbox = [
            #     word_bbox[0],
            #     word_bbox[1],
            #     word_bbox[4],
            #     word_bbox[5],
            # ]
            processed_word_bbox = inscribed_rect_coords(word_bbox)
            word["box"] = processed_word_bbox
            # Check that the word is not out of the img
            word_flag = check_bbox(processed_word_bbox)
            if word_flag:
                words_flag = True

        if segment_flag or words_flag:
            bboxes_out = True

    if bboxes_out:
        b_df.loc[b_df["file_name"] == name, "words_out"] = True
    else:
        b_df.loc[b_df["file_name"] == name, "words_out"] = False

    return postprocessed_data, b_df


def save_postprocessed_data(data: dict, annotation_name: str, saving_data: dict):
    annotation_path = os.path.join(
        ROOT,
        saving_data["language"],
        saving_data["school"],
        "dataset_output",
        "annotations_post",
        annotation_name,
    )
    write_json(data, annotation_path)


def save_blueprint_data(b_df: pd.DataFrame):
    b_df.to_csv(BLUEPRINT, index=False)


def postprocess_annotations(dir: str, b_df: pd.DataFrame, saving_data: dict):
    # Loop over the samples
    for annotation in tqdm(sorted(os.listdir(dir))):
        sample_name = annotation.split(".")[0][:-12]
        annotation_path = os.path.join(dir, annotation)
        annotation_data = read_json(annotation_path)
        postprocessed_annotation_data, b_df = postprocess_bboxes(
            sample_name, annotation_data, b_df
        )
        save_postprocessed_data(postprocessed_annotation_data, annotation, saving_data)

    return b_df


if __name__ == "__main__":
    postprocess()
