from os.path import join, abspath, dirname
from typing import List
import os
import json
from pdf2image import convert_from_bytes
from PyPDF2 import PdfWriter, PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import textwrap


def get_merit_subset_paths(language: str, school: str) -> List:
    root_path = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "data")
    subset_path = join(root_path, "original", language, school, "dataset_output", "annotations")

    file_list = []
    if os.path.exists(subset_path):
        file_list = [join(subset_path, f) for f in os.listdir(subset_path) if os.path.isfile(join(subset_path, f))]

    return file_list


def get_annotation(path: str):
    return read_json(path)


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def clean_annotation(annotation: dict):

    text = ""
    for element in annotation["form"]:
        text += f" {element['text']}"

    return text


def generate_pdf(text: str) -> BytesIO:

    # Create a temporary buffer for the PDF
    buffer = BytesIO()

    # Create the PDF with reportlab
    c = canvas.Canvas(buffer, pagesize=letter)

    # Set the maximum width for the text
    max_width = 500  # Adjust this value as needed
    text_object = c.beginText(100, 750)
    text_object.setFont("Helvetica", 12)

    # Add the text, automatically adjusting
    for line in text.splitlines():
        # Wrap the text using textwrap
        wrapped_lines = textwrap.wrap(line, width=70)  # Adjust width as needed

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


def convert_pdf_to_img(pdf_buffer: BytesIO):
    return convert_from_bytes(pdf_buffer.getvalue())


def generate_img(text: str):
    pdf_bytes = generate_pdf(text)
    img = convert_pdf_to_img(pdf_bytes)

    return img


def save_sample(path: str, img: List):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img[0].save(path, "PNG")
