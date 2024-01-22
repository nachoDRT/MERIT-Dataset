import json
import random
import csv
import copy
import os
from pathlib import Path
import numpy as np


def load_csv(filepath: str):
    """Loads csv file with 1 row per argument"""
    data_fields = []
    with open(filepath, "r") as f:
        csvreader = csv.reader(f, delimiter=" ")
        for row in csvreader:
            data_fields.append(row[0])

    return data_fields


def load_json(filepath: str):
    """Loads any json file"""
    with open(filepath, "r") as json_file:
        data = json_file.read()
    data = json.loads(data)
    return data


def write_json(filepath: str, json_object):
    """Outputs dict to json file"""
    with open(filepath, "w") as outfile:
        outfile.write(json_object)
    return


def return_verbose_grade(grade: int, verbose_level=0):
    if verbose_level == 1:
        grade_text = ""
        if grade < 5:
            grade_text = "Insuficiente"
        elif grade < 6:
            grade_text = "Suficiente"
        elif grade < 7:
            grade_text = "Bien"
        elif grade < 9:
            grade_text = "Notable"
        else:
            grade_text = "Sobresaliente"
        grade_string = "(" + grade_text + " - " + str(grade) + ")"
        return grade_string
    else:
        return str(grade)


class DataLoader:
    """Loads names, surnames and subjects"""

    names = []
    surnames = []
    subjects = {}

    def __init__(self, res_path, language, gender, origin) -> None:
        assets_path = Path(res_path).parent

        # Load names, surnames and subjects
        names_path = os.path.join(
            assets_path,
            origin,
            "first_names",
            "".join([gender, "_first_names_", origin, ".txt"]),
        )
        surnames_path = os.path.join(
            assets_path,
            origin,
            "family_names",
            "".join(["family_names_", origin, ".txt"]),
        )
        subjects_path = os.path.join(
            res_path, "subjects", "".join(["subjects_", language, ".json"])
        )
        self.names = load_csv(names_path)
        self.surnames = load_csv(surnames_path)
        self.subjects_json = load_json(subjects_path)
        self.subjects = self.subjects_json["subjects"]
        # self.academic_year_tags = self.subjects_json["academic_years_tags"]


class Person:
    """Person class with methods to generate random curriculums/names/etc"""

    curriculum = []

    def __init__(
        self,
        res_path: str,
        language: str,
        courses: list = [],
        student: bool = True,
        n_subjects: int = 0,
        gender: str = "",
        origin: str = "",
        student_grades_seeds: dict = {},
    ) -> None:
        self.dataLoader = DataLoader(
            res_path=res_path, language=language, gender=gender, origin=origin
        )
        self._name = random.choice(list(self.dataLoader.names))
        self._first_surname = random.choice(list(self.dataLoader.surnames))
        self._second_surname = random.choice(list(self.dataLoader.surnames))
        self.curriculum = []

        if student:
            self.years = courses
            self.populate_courses(n_subjects, student_grades_seeds)

    def get_full_name(self):
        full_name = self._name + " " + self._first_surname + " " + self._second_surname
        return full_name

    def populate_courses(self, n_subjects: int, student_grades_seeds: dict):
        """Choose subjects randomly and assign grades"""
        if len(self.curriculum) == 0:
            for year_index, year in enumerate(self.years):
                course = []
                subjects = list(self.dataLoader.subjects)
                random.shuffle(subjects)
                subjects = subjects[0 : n_subjects[year_index]]

                for subject in subjects:
                    subject_dict = {}
                    grade = np.random.normal(
                        student_grades_seeds["mean"], student_grades_seeds["dev"], 1
                    )[0]
                    subject_dict["grade"] = round(np.clip(grade, 0, 10))
                    verbose_name = random.choice(
                        list(self.dataLoader.subjects[subject])
                    )
                    subject_dict["verbose_name"] = verbose_name
                    course.append({str(subject + "_" + year): subject_dict})

                self.curriculum.append(course)

    def get_ground_truth(self, year: int, number_subjects: int):
        """Get ground truth for a certain year (requires number of subjects to output)"""
        return_array = copy.deepcopy(self.curriculum[year][0:number_subjects])

        for i, subject in enumerate(return_array):
            for j, s in enumerate(list(return_array[i])):
                return_array[i][s].pop("verbose_name")

        return return_array

    def get_replacements(self, verbose_level=0):
        replacements = {}

        i = 0
        for year in self.curriculum:
            j = 0
            for subject in year:
                key = list(subject)[0]
                replacements[f"replace_subject_{i}_{j}"] = subject[key]["verbose_name"]
                replacements[f"replace_grade_{i}_{j}"] = return_verbose_grade(
                    subject[key]["grade"], verbose_level
                )
                j += 1
            i += 1

        return replacements
