from typing import List, Dict
import os
import json
from pdf2image import convert_from_bytes
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import textwrap
from datasets import load_dataset
from PIL import Image as PÌLIMage
import warnings
from os.path import abspath, dirname, join
import pandas as pd
from datasets.features import Image
import cv2
import numpy as np
from reportlab.pdfbase.pdfmetrics import stringWidth
import copy


PARAGRAPH_DETECTION_THRES = 50
MAX_CHARACTERS = 65


def get_merit_dataset_iterator(subset_name: str, split: str, decode=None):
def get_merit_dataset_iterator(subset_name: str, split: str, decode=True):

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", subset_name, split=split, streaming=True)

    dataset = dataset.cast_column("image", Image(decode=decode))
    dataset_iterator = iter(dataset)

    return dataset_iterator, dataset


    if decode:
        dataset = dataset.cast_column("image", PÌLIMage(decode=False))

    dataset_iterator = iter(dataset)

    return dataset_iterator, dataset


def get_merit_dataset_splits(merit_subset_name):

    _, dataset = get_merit_dataset_iterator(subset_name=merit_subset_name, split=None)
    splits = list(dataset.keys())

    return splits


def get_annotation(path: str):
    return read_json(path)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def clean_line_annotation(annotation: Dict, years: List):

    record = {}
    paragraphs = []
    last_element = None

    for element in annotation["form"]:

        if element["label"] != "other":
            clean = element["label"].split("_answer")

            len_clean = len(clean)
            clean = clean[0]

            if clean not in years:

                if clean not in record:
                    if len_clean == 1:
                        subject_content = {"subject": element["text"], "grade": ""}
                    else:
                        subject_content = {"subject": "", "grade": element["text"]}

                    record[clean] = subject_content

                else:
                    subject_content = record[clean]
                    if len_clean == 1:
                        subject_content["subject"] = element["text"]
                    else:
                        subject_content["grade"] = element["text"]

            else:
                # Flag to insert the grades
                paragraphs.append(f" {element['text']}")
                paragraphs.append(True)

        else:
            new_paragraph = detect_new_paragraph(element, last_element)

            if new_paragraph:
                try:
                    text += ""
                    paragraphs.append(text)
                    text = f" {element['text']}"
                except NameError:
                    text = ""
            else:
                text += f" {element['text']}"

        last_element = element

    text = get_string_from_paragraphs(paragraphs, record)

    return text


def clean_line_annotation_token_class(annotation: Dict):

    text = ""
    # lines = []
    list_of_dicts = []

    for element in annotation["form"]:

        text += f" {element['text']}\n"
        # lines.append(element["text"])

        line_dict = element
        line_dict["box"] = None

        words = []
        for subelement in line_dict["words"]:
            subelement["box"] = None
            words.append(subelement)

        line_dict["words"] = words

        list_of_dicts.append(line_dict)

    form_with_no_boxes = {}
    form_with_no_boxes["form"] = list_of_dicts

    return text, form_with_no_boxes


def detect_new_paragraph(element: Dict, last_element: Dict) -> bool:

    try:
        # Get 'y' coordinate
        y_last = last_element["box"][1]
        y = element["box"][1]
        if y - y_last > PARAGRAPH_DETECTION_THRES:
            new_paragraph = True
        else:
            new_paragraph = False
        return new_paragraph

    except TypeError:
        return True


def get_string_from_paragraphs(paragraphs: List, record: Dict) -> str:

    string = ""

    for element in paragraphs:

        if type(element) is bool:
            string += compose_table_as_string(record)
        else:
            string += element

    return string


def clean_paragraph_annotation(annotation: Dict):

    text = ""
    for element in annotation["form"]:
        text += f" {element['text']}"

    return text


def compose_table_as_string(record: Dict):

    string_table = "\n"

    for instance in record.values():
        string_table += f"{instance['subject']} {instance['grade']}\n"

    return string_table


def generate_pdf(text: str) -> BytesIO:

    # Create a temporary buffer for the PDF
    buffer = BytesIO()

    # Create the PDF with reportlab
    c = canvas.Canvas(buffer, pagesize=letter)

    text_object = c.beginText(100, 750)
    text_object.setFont("Helvetica", 12)

    for line in text.splitlines():
        wrapped_lines = textwrap.wrap(line, width=MAX_CHARACTERS)

        for wrapped_line in wrapped_lines:
            text_object.textLine(wrapped_line)

    c.drawText(text_object)
    c.save()

    # Move buffer position to beginning
    buffer.seek(0)

    # Create PDF writer object
    output = PdfWriter()

    # Add the page
    page = PdfReader(buffer).pages[0]
    output.add_page(page)

    # Write to a temporary buffer
    output_buffer = BytesIO()
    output.write(output_buffer)
    output_buffer.seek(0)

    return output_buffer


def generate_pdf_with_bboxes(text: str):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)

    x_start, y_start = 100, 750
    line_height = 14  # Ajusta según tamaño de fuente
    bounding_boxes = []

    lines_list = []

    y = y_start

    for line in text.splitlines():
        wrapped_lines = textwrap.wrap(line, width=MAX_CHARACTERS)
        for wrapped_line in wrapped_lines:
            line_dict = {}
            words = wrapped_line.split()
            x = x_start
            line_bboxes = []
            for word in words:
                word_width = stringWidth(word, "Helvetica", 12)
                bbox = {"word": word, "bbox": (int(x), int(y - 2), int(x + word_width), int(y + line_height - 2))}
                line_bboxes.append(bbox)
                bounding_boxes.append(bbox)
                c.drawString(x, y, word)
                x += word_width + stringWidth(" ", "Helvetica", 12)  # Añadir espacio
            y -= line_height
            line_dict["box"] = []
            lines_list.append(line_dict)

    c.save()
    buffer.seek(0)

    label_dict = {}
    label_dict["form"] = lines_list

    return buffer, bounding_boxes


def generate_line_pdf_token_class(text: str, form_annotation: Dict) -> BytesIO:

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 8)

    x_start, y_start = 100, 750
    line_height = 10  # Ajusta según tamaño de fuente
    # bounding_boxes = []

    lines_list = []

    y = y_start

    transformed_annotation = copy.deepcopy(form_annotation)

    for i, (annotation_line, line) in enumerate(zip(form_annotation["form"], text.splitlines())):
        # for i, line in enumerate(text.splitlines()):
        wrapped_lines = textwrap.wrap(line, width=MAX_CHARACTERS)

        for wrapped_line in wrapped_lines:
            # print(annotation_line["text"], wrapped_line)

            words = wrapped_line.split()
            x = x_start
            line_bboxes = []
            for j, word in enumerate(annotation_line["words"]):
                word = word["text"]
                word_width = stringWidth(word, "Helvetica", 8)
                bbox = {"word": word, "bbox": (int(x), int(y - 2), int(x + word_width), int(y + line_height - 2))}
                line_bboxes.append(bbox)
                # bounding_boxes.append(bbox)
                c.drawString(x, y, word)
                # x += word_width + stringWidth(" ", "Helvetica", 8)  # Add space
                # print(len(transformed_annotation["form"]), i)
                # print(len(transformed_annotation["form"][i]["words"]), j)
                # print(transformed_annotation["form"][i])
                transformed_annotation["form"][i]["words"][j]["box"] = [
                    int(x),
                    int(y - 2),
                    int(x + word_width),
                    int(y + line_height - 2),
                ]
            y -= line_height
            # line_dict["box"] = []
            # lines_list.append(line_dict)

        # print(line_bboxes)
        # print(line_bboxes)
        transformed_annotation["form"][i]["box"] = [
            line_bboxes[0]["bbox"][0],
            line_bboxes[0]["bbox"][1],
            line_bboxes[-1]["bbox"][2],
            line_bboxes[-1]["bbox"][3],
        ]

    c.save()
    buffer.seek(0)

    label_dict = {}
    label_dict["form"] = lines_list

    return buffer, transformed_annotation


def generate_line_pdf(text: str) -> BytesIO:

    # Create a temporary buffer for the PDF
    buffer = BytesIO()

    # Create the PDF with reportlab
    c = canvas.Canvas(buffer, pagesize=letter)

    text_object = c.beginText(35, 750)
    text_object.setFont("Helvetica", 8)

    for line in text.splitlines():
        wrapped_lines = textwrap.wrap(line, width=150)
        for i, wrapped_line in enumerate(wrapped_lines):
            text_object.textLine(wrapped_line)

        # TODO useful for next degradation
        # text_object.textLine(" ")

    c.drawText(text_object)
    c.save()

    # Move buffer position to beginning
    buffer.seek(0)

    return buffer


def convert_pdf_to_img(pdf_buffer: BytesIO):
    return convert_from_bytes(pdf_buffer.getvalue())


def generate_img(text: str):
    pdf_bytes = generate_pdf(text)
    img = convert_pdf_to_img(pdf_bytes)

    return img


def generate_img_with_bboxes(text: str):
    pdf_bytes, bboxes = generate_pdf_with_bboxes(text)
    img = convert_pdf_to_img(pdf_bytes)
    return img, bboxes


def generate_line_img(text: str):
    pdf_bytes = generate_line_pdf(text)
    img = convert_pdf_to_img(pdf_bytes)

    return img


def generate_line_img_token_class(text: str, form_annotation: Dict):
    pdf_bytes, form_annotation = generate_line_pdf_token_class(text, form_annotation)
    img = convert_pdf_to_img(pdf_bytes)

    return img, form_annotation


def save_sample(path: str, img: List):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img[0].save(path, "PNG")


def get_sample_data(sample):

    img = sample["image"]
    annotation = json.loads(sample["ground_truth"])

    return img, annotation


def get_year_from_subject(subject: str, years: List) -> str:

    for year in years:
        if year in subject:
            return year

    warnings.warn(f"Unable to extract the academic year from {subject}")

    return None


def extract_key_annotations(dataset_data: Dict, academic_years: List):
    years = []
    subjects = []
    grades = {}

    for segment in dataset_data["form"]:
        if segment["label"] == "other":
            pass

        # Look for academic years
        elif segment["label"] in academic_years:
            years.append(segment["label"])

        # Look for grades
        elif segment["label"].split("_")[-1] == "answer":
            grade = segment["text"]
            subject = segment["label"][:-7]
            grades[subject] = grade

        # Otherwise it is a subject
        else:
            subject_dict = {}
            subject_dict[segment["label"]] = segment["text"]
            subject_dict["year"] = get_year_from_subject(segment["label"], academic_years)
            subjects.append(subject_dict)

    ground_truth = process_extractions(years, subjects, grades)

    return ground_truth


def process_extractions(years: List, subjects: List, grades: Dict) -> Dict:
    ground_truth = {}

    for year in years:
        ground_truth_year = []
        for subject in subjects:
            if subject["year"] == year:
                subject_dict = {}
                subject_dict["subject"] = list(subject.values())[0]
                subject_dict["grade"] = grades[list(subject.keys())[0]]
                ground_truth_year.append(subject_dict)
        ground_truth[year] = ground_truth_year

    return ground_truth


def format_annotations_cordv2_style(funsd_format_data, academic_years: List) -> Dict:

    cordv2_format_data = {}

    cordv2_format_data = extract_key_annotations(funsd_format_data, academic_years)
    cordv2_format_data = {"gt_parse": cordv2_format_data}

    return cordv2_format_data


def read_dataset_features_json():
    config_path = join(dirname(dirname(abspath(__file__))), "config", "dataset_features.json")
    dataset_features = read_json(config_path)
    return dataset_features

def load_watermarks():

    watermarks = {}

    if language == "es":
        schools = [
            "aletamar",
            "britanico",
            "deus",
            "liceo",
            "lusitano",
            "monterraso",
            "patria",
        ]
    elif language == "en":
        schools = [
            "freefields",
            "greenfields",
            "james",
            "paloalto",
            "pinnacle",
            "salesianum",
            "whitney",
        ]

    watermarks_paths_root = join(dirname(abspath(__file__)), "assets")

    for school in schools:
        watermark_file_name = f"watermark_{school}.png"
        watermark_path = join(watermarks_paths_root, watermark_file_name)
        watermark_pil_img = PÌLIMage.open(watermark_path)
        np_img = np.array(watermark_pil_img)
        watermark_cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGRA)
        watermarks[school] = watermark_cv_img

    return watermarks


def add_transparent_image(background, foreground, alpha_factor=1.0, x_offset=None, y_offset=None):
    """
    Function sourced from StackOverflow contributor Ben.

    This function was found on StackOverflow and is the work of Ben, a contributor
    to the community. We are thankful for Ben's assistance by providing this useful
    method.

    Original Source:
    https://stackoverflow.com/questions/40895785/
    using-opencv-to-overlay-transparent-image-onto-another-image
    """

    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert fg_channels == 4, f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y : fg_y + h, fg_x : fg_x + w]
    background_subsection = background[bg_y : bg_y + h, bg_x : bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255 * alpha_factor  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite

    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = PÌLIMage.fromarray(background)

    return background
