from typing import List
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


def get_merit_dataset_iterator(subset_name: str, split: str, decode=None):

    print("Loading Dataset")

    dataset = load_dataset("de-Rodrigo/merit", subset_name, split=split, streaming=True)

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


def get_sample_data(sample):

    img = sample["image"]
    annotation = json.loads(sample["ground_truth"])

    return img, annotation
