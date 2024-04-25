import person_generator
import zipfile
import os
import docx2pdf
import platform
from pathlib import Path
import subprocess
from pdf2image import convert_from_path
import numpy as np
import PIL
import cv2
import copy
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
from typing import Union

ASSET_ROT = 20
MAPS_THRESHOLD = 15
DEBUG_XML_OUT = False


def check_directories(paths: dict):
    """Check if directories exist, if not create them"""
    if not os.path.exists(paths["template_path"]):
        print("ERROR, create template first!")

    if not os.path.exists(paths["res_path"]):
        print("ERROR, no resources!")

    if not os.path.exists(paths["ds_path"]):
        os.makedirs(paths["ds_path"])

    if not os.path.exists(paths["annotations_path"]):
        os.makedirs(paths["annotations_path"])

    if not os.path.exists(paths["images_path"]):
        os.makedirs(paths["images_path"])

    if not os.path.exists(paths["pdf_path"]):
        os.makedirs(paths["pdf_path"])


def compute_pos_from_map(map: np.array):
    probs = np.where(map > MAPS_THRESHOLD, map / 255.0, 0)

    total_prob = np.sum(probs)
    if total_prob > 0:
        norm_probs = probs / total_prob
    else:
        raise ValueError("The given map is not valid")

    choice = np.random.choice(np.arange(probs.size), p=norm_probs.flatten())

    x, y, _ = np.unravel_index(choice, probs.shape)

    # TODO Maps are positioned a little too low. As a temporal mesurement, we offset
    # the 'y' coordinate. Alternatives: modify the transparent superposition method
    # 'add_transparent_image()' or modify all the signature and stamping maps
    y -= 300

    return x, y


def test_map(map: np.array, show: bool = True, n: int = 10):
    factor = 0.25
    map_copy = copy.deepcopy(map)
    map_copy = resize_asset(map_copy, factor=factor)

    print("Testing map")
    for i in tqdm(range(n)):
        x, y = compute_pos_from_map(map)
        # Ajustar las coordenadas para el mapa redimensionado
        cv2.circle(
            map_copy,
            (
                int(y * factor),
                int(x * factor),
            ),  # Asegúrate de que las coordenadas están en el orden correcto
            radius=2,
            color=(0, 0, 255),
            thickness=2,
        )

    if show:
        cv2.imshow("Map test result", map_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def sign_document(signature, map, doc_img):
    y_offset, x_offset = compute_pos_from_map(map)

    signature = resize_asset(signature, factor=random.uniform(0.55, 1))
    signature = rotate_asset(signature)

    add_transparent_image(doc_img, signature, x_offset=x_offset, y_offset=y_offset)

    return doc_img


def stamp_document(stamp, map, doc_img):
    y_offset, x_offset = compute_pos_from_map(map)
    stamp = resize_asset(stamp, factor=0.25)
    stamp = rotate_asset(stamp)

    add_transparent_image(
        doc_img, stamp, alpha_factor=0.6, x_offset=x_offset, y_offset=y_offset
    )

    return doc_img


def resize_asset(asset, factor):
    original_height, original_width, _ = asset.shape

    new_height = int(original_height * factor)
    new_width = int(original_width * factor)

    resized_asset = cv2.resize(asset, (new_width, new_height))

    return resized_asset


def rotate_asset(asset):
    height, width = asset.shape[:2]
    center = (width / 2, height / 2)
    angle = random.uniform(-ASSET_ROT, ASSET_ROT)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_asset = cv2.warpAffine(asset, rotation_matrix, (width, height))

    return rotated_asset


def add_transparent_image(
    background, foreground, alpha_factor=1.0, x_offset=None, y_offset=None
):
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

    assert (
        bg_channels == 3
    ), f"background image should have exactly 3 channels (RGB). found:{bg_channels}"
    assert (
        fg_channels == 4
    ), f"foreground image should have exactly 4 channels (RGBA). found:{fg_channels}"

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
    foreground_colors = cv2.cvtColor(foreground_colors, cv2.COLOR_BGR2RGB)
    alpha_channel = foreground[:, :, 3] / 255 * alpha_factor  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = (
        background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask
    )

    # overwrite the section of the background image that has been updated
    background[bg_y : bg_y + h, bg_x : bg_x + w] = composite


def debug_xml_out(path: str):
    # Namespace generalmente utilizado en documentos DOCX
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    with zipfile.ZipFile(path, "r") as zin:
        # Leer el contenido del archivo document.xml
        xml_content = zin.read("word/document.xml").decode("utf-8")

        # Parsear el contenido XML
        root = ET.ElementTree(ET.fromstring(xml_content)).getroot()

        # Extraer y mostrar el texto plano por segmento
        for elem in root.iter(namespace + "t"):
            print("Segmento de texto:", elem.text)


class SchoolRecord:
    """Allows for pdf-png-annotations creation"""

    def __init__(
        self,
        id: int,
        paths: dict,
        props: dict,
        reqs: dict,
        lang: str,
        gender: str,
        origin: str,
        grades_seeds: dict,
        courses: list,
        n_subjects: int,
        retrieved_info: dict,
        verbose_level: int = 0,
        system_alpha_grades: dict = {},
        grade_synonyms: Union[bool, int] = False,
        grade_decimals: bool = False,
        alpha_numeric_separator: str = "",
        new_school: bool = False,
        new_student: bool = False,
    ) -> None:
        self.paths = paths
        self.props = props
        self.reqs = reqs
        self.new_school = new_school
        self.new_student = new_student

        if self.new_school:
            genders = ["female", "male"]
            secre_gender = random.choice(genders)
            head_gender = random.choice(genders)

            student_gender = gender
            student_origin = origin
            student_grades_seeds = grades_seeds

            self.secre = person_generator.Person(
                res_path=self.paths["res_path"],
                language=lang,
                student=False,
                gender=secre_gender,
                origin=lang,
            )
            self.director = person_generator.Person(
                res_path=paths["res_path"],
                language=lang,
                student=False,
                gender=head_gender,
                origin=lang,
            )
            self.student = person_generator.Person(
                res_path=self.paths["res_path"],
                language=lang,
                requirements=self.reqs,
                grade_decimals=grade_decimals,
                courses=courses,
                student=True,
                n_subjects=n_subjects,
                gender=student_gender,
                origin=student_origin,
                student_grades_seeds=student_grades_seeds,
            )

        else:
            self.secre = retrieved_info["secretary"]
            self.director = retrieved_info["director"]
            if self.new_student or retrieved_info["re_spawn_student"]:
                if retrieved_info["re_spawn_student"]:
                    student_gender = retrieved_info["student_gender"]
                    student_origin = retrieved_info["student_name_origin"]
                    student_grades_seeds = retrieved_info["average_grade"]
                else:
                    student_gender = gender
                    student_origin = origin
                    student_grades_seeds = grades_seeds

                self.student = person_generator.Person(
                    res_path=self.paths["res_path"],
                    language=lang,
                    requirements=self.reqs,
                    grade_decimals=grade_decimals,
                    courses=courses,
                    student=True,
                    n_subjects=n_subjects,
                    gender=student_gender,
                    origin=student_origin,
                    student_grades_seeds=student_grades_seeds,
                )
            else:
                self.student = retrieved_info["student_object"]

        self.set_replacements(
            system_alpha_grades, grade_synonyms, verbose_level, alpha_numeric_separator
        )
        id_parts = id.split("_")
        id = "_".join(id_parts[:-1])
        self.id = id

        self.pdf_path = os.path.join(
            self.paths["pdf_path"],
            str(id) + ".pdf",
        )

        check_directories(paths=self.paths)

    def get_info_retrieval(self):
        info_retrieval = {}
        if self.new_school:
            info_retrieval["director"] = self.director.get_full_name()
            info_retrieval["secretary"] = self.secre.get_full_name()
        else:
            info_retrieval["director"] = self.director
            info_retrieval["secretary"] = self.secre

        info_retrieval["student_object"] = self.student
        info_retrieval["student_name"] = self.student.get_full_name()
        info_retrieval["re_spawn_student"] = False

        return info_retrieval

    def set_replacements(
        self,
        system_alpha_grades: dict,
        grade_synonyms: Union[bool, int] = False,
        verbose_level: int = 0,
        alpha_numeric_separator: str = "",
    ):
        """Populates replacements dictionary"""
        # if self.new_student or self.new_school:
        self.replacements = self.student.get_replacements(
            system_alpha_grades, grade_synonyms, verbose_level, alpha_numeric_separator
        )
        if self.new_school:
            self.replacements["replace_director"] = self.director.get_full_name()
            self.replacements["replace_secre"] = self.secre.get_full_name()
        else:
            self.replacements["replace_director"] = self.director
            self.replacements["replace_secre"] = self.secre

        self.replacements["replace_alumno"] = self.student.get_full_name()

    def docx_replace(self, old_file, new_file, rep):
        """Replaces docx xml according to rep (replacements dictionary)"""
        zin = zipfile.ZipFile(old_file, "r")
        zout = zipfile.ZipFile(new_file, "w")
        for item in zin.infolist():
            buffer = zin.read(item.filename)
            if item.filename == "word/document.xml":
                res = buffer.decode("utf-8")
                for r in rep:
                    if any(char.isdigit() for char in r):
                        # If there is a number only replace once
                        res = res.replace(r, rep[r], 1)
                    else:
                        # For all other cases replace all instances
                        res = res.replace(r, rep[r])
                buffer = res.encode("utf-8")
                words = res.split()
            zout.writestr(item, buffer)

        zout.close()
        zin.close()

        if DEBUG_XML_OUT:
            debug_xml_out(new_file)

    def create_pdf(self):
        """Works over temporary output.docx file and outputs pdf"""
        intermediate_path = os.path.join(self.paths["base_path"], "output.docx")

        # Word replacement & PDF generation
        self.docx_replace(
            self.paths["template_path"], intermediate_path, self.replacements
        )

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
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Remove the docx
        os.remove(new_file_path)

        return self.pdf_path

    def create_pngs(self, assets: dict):
        """Creates png from pdf (returns png paths)"""
        # PDF to PNG
        images = convert_from_path(self.pdf_path)

        png_paths = []

        for i, image in enumerate(images):
            # test_map(assets["stamp_map"], n=100)
            if type(image) is PIL.PpmImagePlugin.PpmImageFile:
                image = np.array(image)

            image = stamp_document(assets["stamp"], assets["stamp_map"], image)
            image = sign_document(assets["signature"], assets["signature_map"], image)

            if type(image) is np.ndarray:
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(image)

            png_filename = (
                str(self.id).zfill(self.props["doc_name_zeros_fill"])
                + "_"
                + str(i)
                + ".png"
            )
            image_path = os.path.join(self.paths["images_path"], png_filename)

            # dict = list_of_dicts[i]
            png_paths.append(image_path)

            try:
                image.save(image_path)

            except FileNotFoundError:
                os.makedirs(self.paths["images_path"])
                image.save(image_path)
        return png_paths
