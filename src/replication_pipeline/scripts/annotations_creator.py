import os
import json
import copy
from fpdf import FPDF
import copy
from unidecode import unidecode
from pathlib import Path
from typing import Iterable, Any
from pdfminer.high_level import extract_pages
from person_generator import return_verbose_grade
from itertools import chain
import cv2
import numpy as np

global_annotations = {}

SEGMENT_DEPTH_LEVEL = 2
CHECK_BBOXES = True


def transform_bbox(bbox: list, w1, w2, h1, h2, invertY: False) -> list:
    """Transform bbox from PDF to PNG coordinates"""
    scaler = w2 / w1
    new_box = []

    for coordinate in bbox:
        new_box.append(int(int(coordinate) * scaler))

    if invertY:
        y1 = new_box[1]
        y2 = new_box[3]
        new_box[1] = h2 - y2
        new_box[3] = h2 - y1

    return new_box


def get_bbox(o: Any) -> str:
    """Bounding box of LTItem if available (transformed to PNG coordinates)"""

    W_1 = 595.2755
    W_2 = 1657
    H_1 = 841.8898
    H_2 = 2339

    if hasattr(o, "bbox"):
        bbox = []
        for i in o.bbox:
            bbox.append(i)

        bbox = transform_bbox(bbox, W_1, W_2, H_1, H_2, invertY=True)

        return bbox
    else:
        return ""


def get_optional_text(o: Any) -> str:
    """Text of LTItem if available, otherwise empty string"""
    if hasattr(o, "get_text"):
        return o.get_text().strip()
    else:
        return ""


def get_individual_words(segment):
    """Returns a words dictionary for a given segment"""
    words = []

    while True:
        word_dict = {}
        word = ""
        end_of_segment = False

        W_1 = 595.2755
        W_2 = 1657
        H_1 = 841.8898
        H_2 = 2339

        x1 = 0
        y1 = 0

        for c in segment:
            if c.__class__.__name__ in ["LTAnno"]:
                if len(word) > 0:
                    word_box = [x1, y1, x2, y2]

                    word_dict["box"] = word_box
                    word_dict["text"] = word
                    words.append(copy.deepcopy(word_dict))

                end_of_segment = True
                break

            character = get_optional_text(c)
            if character != "":
                word += character
                if len(word) == 1:
                    bbox = get_bbox(c)
                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = bbox[2]
                    y2 = bbox[3]

                elif len(word) > 1:
                    # try:
                    bbox = get_bbox(c)
                    x2 = bbox[2]
                    y2 = bbox[3]

                    # except UnboundLocalError:
                    #     bbox = get_bbox(c)
                    #     x1 = bbox[0]
                    #     y1 = bbox[1]
                    #     x2 = bbox[2]
                    #     y2 = bbox[3]
            else:
                bbox = get_bbox(c)
                x2 = bbox[2]
                y2 = bbox[3]

                word_box = [x1, y1, x2, y2]

                if len(word) > 0:
                    word_dict["box"] = word_box
                    word_dict["text"] = word
                    words.append(copy.deepcopy(word_dict))

                word = ""
                continue

        if end_of_segment:
            break

    return words


def create_annotations_recursively(o: Any, depth=0):
    """Return annotations based on pdfminer view of a single pdf view"""
    # print(f'{depth} {o.__class__.__name__}')
    global global_annotations
    global count

    if depth == 0:
        count = 0
        global_annotations["form"] = []

    if depth == SEGMENT_DEPTH_LEVEL and o.__class__.__name__ == "LTTextLineHorizontal":
        if hasattr(o, "get_text"):
            new_annotation = {}
            new_annotation["box"] = get_bbox(o)
            new_annotation["text"] = get_optional_text(o)
            new_annotation["label"] = "other"
            new_annotation["words"] = get_individual_words(o)
            new_annotation["linking"] = []
            new_annotation["id"] = count
            global_annotations["form"].append(new_annotation)
            count += 1

    # Next level down
    if isinstance(o, Iterable):
        for i in o:
            create_annotations_recursively(i, depth=depth + 1)

    if depth == 0:
        return copy.deepcopy(global_annotations)


def update_course_tags(
    course_curriculum,
    annotations_page,
    verbose_level: int,
    sample_name: str = None,
    page: int = None,
    max_subjects: int = 15,
):
    success = 1

    # for subject in course_curriculum[0:max_subjects]:
    for subject in course_curriculum:
        tag = list(subject.keys())[0]
        verbose_name = subject[tag]["verbose_name"]
        subject_grade = subject[tag]["grade"]
        found = 0

        # Search in pdf
        for segment in annotations_page:
            segment_text = segment["text"]
            segment_tag = segment["label"]

            check1 = unidecode(verbose_name) == unidecode(segment_text)
            check2 = segment_tag == "other"

            if check1 and check2:
                found = update_answer_tag(
                    annotations_page, tag, subject_grade, verbose_level
                )
                segment["label"] = tag
                break

        if found:
            continue

        if not found:
            print(
                "Not found: "
                + unidecode(verbose_name)
                + " "
                + tag
                + " "
                + sample_name
                + " "
                + str(page)
            )
            success = 0
    return success


def update_answer_tag(annotations_page, tag, subject_grade, verbose_level: int):
    found = 0
    subject_grade_verbose = return_verbose_grade(subject_grade, verbose_level)

    for segment in annotations_page:
        segment_text = segment["text"]
        segment_tag = segment["label"]

        check1 = True
        check2 = segment_tag == "other"

        if check1 and check2:
            check3 = subject_grade_verbose == segment_text

            if check3:
                segment["label"] = tag + "_answer"
                found = 1
                break
    return found


def school_nickname_to_key(langs_schools: dict, nickname: str):
    for school_key, school_nickname in langs_schools.items():
        if nickname == school_nickname["nickname"]:
            return school_key

    return None


def get_courses_names(template_layout: dict):
    courses_names = []
    for page_features in template_layout.values():
        courses_names.extend(page_features["academic_years"])

    return courses_names


def get_num_courses_in_page(template_layout: dict) -> int:
    num_courses_in_page = []

    for page_info in template_layout.values():
        num_courses_in_page.append(len(page_info["academic_years"]))

    return num_courses_in_page


def draw_sample_bboxes(page_annotations: dict, page_img: np.array, saving_path: str):
    overlay = np.zeros_like(page_img)

    for segment in page_annotations["form"]:
        for word in segment["words"]:
            rectangle = []
            bbox = word["box"]
            rectangle.extend(bbox[0:2])
            rectangle.append(bbox[2])
            rectangle.append(bbox[1])
            rectangle.extend(bbox[2:4])
            rectangle.append(bbox[0])
            rectangle.append(bbox[3])
            rectangle.extend(bbox[0:2])

            rectangle = np.array(rectangle, dtype=np.int32).reshape((-1, 2))

            cv2.fillPoly(overlay, [rectangle], color=(255, 0, 0))

    # Blend images
    alpha = 0.5
    beta = 1 - alpha
    output_img = cv2.addWeighted(overlay, alpha, page_img, beta, 0)

    cv2.imwrite(saving_path, output_img)
    # cv2.imshow("bboxes", output_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


class AnnotationsCreator:
    def __init__(
        self,
        pdf_path: str,
        paths: dict,
        props: dict,
        reqs: dict,
        lang: str,
        school: str,
    ):
        self.paths = paths
        self.pdf_file_path = pdf_path
        self.props = props
        # self.courses_names = self.props["courses_names"]
        self.reqs = reqs
        self.school_i = school_nickname_to_key(self.reqs["samples"][lang], school)
        self.possible_courses = self.reqs["samples"][lang][self.school_i][
            "possible_courses_global"
        ]
        self.courses_names = get_courses_names(
            self.reqs["samples"][lang][self.school_i]["template_layout"]
        )
        self.annotations = []

    def create_annotations(self, pdf_path: str, png_paths: list):
        """Create annotations for a given pdf file"""
        pages = extract_pages(pdf_path)
        self.annotations = []
        for page, png_path in zip(pages, png_paths):
            page_annotations = create_annotations_recursively(page)
            self.annotations.append(page_annotations)

            if CHECK_BBOXES:
                sample_img = cv2.imread(png_path)

                check_bboxes_dir = os.path.dirname(os.path.dirname(png_path))
                check_bboxes_dir = os.path.join(check_bboxes_dir, "debug_bboxes")

                if not os.path.isdir(check_bboxes_dir):
                    os.makedirs(check_bboxes_dir)

                sample_file_name = os.path.basename(png_path).split(".")[0]
                sample_file_name = "".join([sample_file_name, "_bbox_debug.png"])
                sample_bboxes_file_path = os.path.join(
                    check_bboxes_dir, sample_file_name
                )

                draw_sample_bboxes(
                    page_annotations, sample_img, sample_bboxes_file_path
                )

        return self.annotations

    def dump_annotations_json(self, png_paths: list):
        """Dump annotations to json file"""
        for i in range(len(png_paths)):
            png_path = png_paths[i]
            # Get xxxx_x part of png file
            annotations_filename = os.path.basename(png_path)[:-4] + ".json"
            annotations_full_path = os.path.join(
                self.paths["annotations_path"], annotations_filename
            )

            with open(annotations_full_path, "w") as outfile:
                json.dump(self.annotations[i], outfile, indent=4)

    def old_new_update_subject_grades_tags(
        self, curriculum: list, n_subjects: int, lang: str, school: str
    ):
        success = 1
        for i, course_curriculum in enumerate(curriculum):
            school_i = school_nickname_to_key(self.reqs["samples"][lang], school)
            s = update_course_tags(
                course_curriculum,
                self.annotations[i]["form"],
                self.reqs["samples"][lang][school_i]["verbose_level"],
                max_subjects=n_subjects[i],
            )
            success *= s

        return success

    def new_update_subject_grades_tags(
        self, curriculum: list, n_subjects: int, lang: str, school: str
    ):
        school_i = school_nickname_to_key(self.reqs["samples"][lang], school)
        num_courses_in_page = get_num_courses_in_page(
            self.reqs["samples"][lang][school_i]["template_layout"]
        )
        success = 1

        curriculum_index = 0
        result = []

        for i in range(len(num_courses_in_page)):
            # Extend the curriculum if needed
            aux = []
            course_curriculum = curriculum[
                curriculum_index : curriculum_index + num_courses_in_page[i]
            ]
            # result.extend(chain(*course_curriculum))
            aux.extend(chain(*course_curriculum))
            result.append(aux)
            curriculum_index += num_courses_in_page[i]

        # for i in range(len(num_courses_in_page)):
        #     print("Hola", i)
        #     # Extend the curriculum if needed
        #     course_curriculum = curriculum[
        #         curriculum_index : curriculum_index + num_courses_in_page[i]
        #     ]
        #     course_curriculum.extend(chain(*course_curriculum))
        #     curriculum_index += num_courses_in_page[i]

        #     what_we_send = course_curriculum[0]

        #     s = update_course_tags(
        #         what_we_send,
        #         self.annotations[i]["form"],
        #         self.reqs["samples"][lang][school_i]["verbose_level"],
        #         self.pdf_file_path,
        #         i,
        #         max_subjects=n_subjects[i],
        #     )
        #     success *= s

        for i in range(len(num_courses_in_page)):
            what_we_send = result[i]

            s = update_course_tags(
                what_we_send,
                self.annotations[i]["form"],
                self.reqs["samples"][lang][school_i]["verbose_level"],
                self.pdf_file_path,
                i,
                max_subjects=n_subjects[i],
            )
            success *= s

        return success

    # TODO The following method might be deprecated
    def update_subject_grades_tags(self, curriculum: list, courses_pages_array: list):
        success = 1
        for i, page in enumerate(courses_pages_array):
            for j, course in enumerate(page):
                # print(f'page: {i}, course: {course[0]}, max_subjects: {course[1]}')
                # Tag subjects and grades
                s = update_course_tags(
                    curriculum[course[0]],
                    self.annotations[i]["form"],
                    self.reqs["verbose_level"],
                    max_subjects=course[1],
                )
                success = success * s
        return success

    def update_annotations_tags_courses(self):
        found = 0

        for i in range(len(self.possible_courses)):
            course_name = self.possible_courses[i]

            for page in self.annotations:
                for segment in page["form"]:
                    segment_text = segment["text"]
                    segment_tag = segment["label"]

                    check1 = course_name in segment_text
                    check2 = segment_tag == "other"

                    if check1 and check2:
                        segment["label"] = self.courses_names[i]
                        found += 1
                        break

        if found != 3:
            print("Some course was not tagged!")
        return found

    def sort_annotations(self):
        annotations = self.annotations
        for page in annotations:
            start = 0
            while True:
                for i in range(start, len(page["form"])):
                    # Find subjects
                    segment = page["form"][i]
                    subject_tag = segment["label"]
                    courses = self.courses_names
                    is_subject = (
                        subject_tag != "other"
                        and subject_tag not in courses
                        and subject_tag[-7:] != "_answer"
                    )

                    if is_subject:
                        start = i + 1
                        # Find grade
                        found = False
                        while (not found) and (i + 1 < len(page["form"])):
                            i = i + 1
                            segment_2 = page["form"][i]
                            segment_2["id"] = i + 1  # Update all ids until grade found

                            if segment_2["label"] == subject_tag + "_answer":
                                # Update grade id
                                found = True
                                grade = segment_2["text"]
                                segment_2["id"] = (
                                    segment["id"] + 1
                                )  # Update grade id to subject id +1
                                break

                        # Sort form by id
                        page["form"].sort(key=lambda x: x["id"])
                        break
                if i == len(page["form"]) - 1:
                    break


class AnnotationsView:
    def __init__(self, annotations: list, pdf_file_path: str, reqs: dict):
        self.annotations = annotations
        self.pdf_file_path = pdf_file_path
        self.font_path = reqs["font_path"]

    def output_annotations_view(self):
        """"""

        W_1 = 1657
        W_2 = 210
        H_1 = 2339
        H_2 = 297
        scaler = W_2 / W_1

        pdf = FPDF()
        pdf.add_font("Arial", "", self.font_path)

        for annotations_page in self.annotations:
            pdf.add_page()
            pdf.set_font("Arial", "B", 6)
            pdf.set_margins(0, 0, 0)
            pdf.set_auto_page_break(False)

            for segment in annotations_page["form"]:
                box = segment["box"]
                text = unidecode(segment["text"])
                label = segment["label"]

                width = (box[2] - box[0]) * scaler
                height = (box[3] - box[1]) * scaler
                x = box[0] * scaler
                y = box[1] * scaler

                pdf.set_xy(x, y)
                if label == "other":
                    border = "B"
                elif label[-7:] == "_answer":
                    border = 1
                else:
                    border = "RLB"

                pdf.cell(border=border, txt=text, w=width, h=height, align="C")

        pdf_output_path = self.pdf_file_path[:-4] + "_annotations.pdf"

        pdf.output(pdf_output_path)
        pass

    def output_annotations_words_view(self):
        W_1 = 1657
        W_2 = 210
        H_1 = 2339
        H_2 = 297
        scaler = W_2 / W_1

        pdf = FPDF()

        for annotations_page in self.annotations:
            pdf.add_page()
            pdf.set_font("Arial", "B", 6)
            pdf.set_margins(0, 0, 0)
            pdf.set_auto_page_break(False)

            for segment in annotations_page["form"]:
                for word in segment["words"]:
                    box = word["box"]
                    text = unidecode(word["text"])

                    width = (box[2] - box[0]) * scaler
                    height = (box[3] - box[1]) * scaler
                    x = box[0] * scaler
                    y = box[1] * scaler

                    # print(f'x: {x}, y: {y}, width: {width}, height: {height}, text: {text}, label: {label}')

                    pdf.set_xy(x, y)
                    pdf.cell(border=True, txt=text, w=width, h=height, align="C")

        pdf_output_path = self.pdf_file_path[:-4] + "_annotations_words.pdf"

        pdf.output(pdf_output_path)
        pass
