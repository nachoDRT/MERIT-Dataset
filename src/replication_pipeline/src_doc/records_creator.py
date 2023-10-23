import person_generator
import zipfile
import os
import docx2pdf
import platform
from pathlib import Path
import subprocess
from pdf2image import convert_from_path
import numpy as np
import cv2
from config import *


def check_directories():
    """Check if directories exist, if not create them"""
    if not os.path.exists(base_path):
        print("ERROR, create template first!!")

    if not os.path.exists(res_path):
        print("ERROR, no resources!!")

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)


class SchoolRecord:
    """Allows for pdf-png-annotations creation"""

    def __init__(self, id: int, parent_dir: str, verbose_level=0) -> None:
        self.student = person_generator.Person(student=True)
        self.secre = person_generator.Person(student=False)
        self.director = person_generator.Person(student=False)

        self.set_replacements(verbose_level)
        self.id = id

        self.pdf_path = os.path.join(pdf_path, str(id).zfill(DOCS_Z_FILL) + ".pdf")

        check_directories()

    def set_replacements(self, verbose_level=0):
        """Populates replacements dictionary"""
        self.replacements = self.student.get_replacements(verbose_level)
        self.replacements["replace_alumno"] = self.student.get_full_name()
        self.replacements["replace_secre"] = self.secre.get_full_name()
        self.replacements["replace_director"] = self.director.get_full_name()

    def docx_replace(self, old_file, new_file, rep):
        """Replaces docx xml according to rep (replacements dictionary)"""
        zin = zipfile.ZipFile(old_file, "r")
        zout = zipfile.ZipFile(new_file, "w")
        for item in zin.infolist():
            buffer = zin.read(item.filename)
            if item.filename == "word/document.xml":
                res = buffer.decode("utf-8")
                # print(res)
                for r in rep:
                    if any(char.isdigit() for char in r):
                        # If there is a number only replace once
                        res = res.replace(r, rep[r], 1)
                    else:
                        # For all other cases replace all instances
                        res = res.replace(r, rep[r])
                buffer = res.encode("utf-8")
            zout.writestr(item, buffer)
        zout.close()
        zin.close()

    def create_pdf(self):
        """Works over temporary output.docx file and outputs pdf"""
        intermediate_path = os.path.join(base_path, "output.docx")

        # Word replacement & PDF generation
        self.docx_replace(template_path, intermediate_path, self.replacements)

        os_name = platform.system()

        if os_name == "Windows":
            docx2pdf.convert(intermediate_path, self.pdf_path, keep_active=True)

        elif os_name == "Linux":
            new_file_name = "".join([os.path.basename(self.pdf_path)[:-3], "docx"])
            new_file_path = os.path.join(Path(intermediate_path).parent, new_file_name)
            os.rename(intermediate_path, new_file_path)
            save_here = Path(self.pdf_path).parent

            subprocess.call(
                [
                    "lowriter",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    save_here,
                    new_file_path,
                ]
            )

        return self.pdf_path

    def create_pngs(self):
        """Creates png from pdf (returns png paths)"""
        # PDF to PNG
        images = convert_from_path(self.pdf_path)

        png_paths = []

        for i, image in enumerate(images):
            image_copy = np.asarray(image)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            png_filename = (
                SCHOOL_NICKNAME
                + "_"
                + str(self.id).zfill(DOCS_Z_FILL)
                + "_"
                + str(i)
                + ".png"
            )
            image_path = os.path.join(images_path, png_filename)

            # dict = list_of_dicts[i]
            png_paths.append(image_path)

            try:
                image.save(image_path)

            except FileNotFoundError:
                os.makedirs(images_path)
                image.save(image_path)
        return png_paths
