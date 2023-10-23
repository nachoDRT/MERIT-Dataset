import json
import random
import csv
import copy

# from config import res_path


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

    def __init__(self, res_path) -> None:
        # Load names, surnames and subjects
        self.names = load_csv(res_path + "/first_names/first_names_spanish.txt")
        self.surnames = load_csv(res_path + "/family_names/family_names_spanish.txt")
        self.subjects_json = load_json(res_path + "/subjects/subjects_spanish.json")
        self.subjects = self.subjects_json["subjects"]
        self.academic_year_tags = self.subjects_json["academic_years_tags"]


class Person:
    """Person class with methods to generate random curriculums/names/etc"""

    curriculum = []  #
    N_subjects = 15  # More subjects are generated than required

    def __init__(self, res_path: str, student: bool = True) -> None:
        self.dataLoader = DataLoader(res_path=res_path)
        self._name = random.choice(list(self.dataLoader.names))
        self._first_surname = random.choice(list(self.dataLoader.surnames))
        self._second_surname = random.choice(list(self.dataLoader.surnames))
        self.curriculum = []

        if student:
            self.years = self.dataLoader.academic_year_tags
            self.populate_courses()

    def get_full_name(self):
        full_name = self._name + " " + self._first_surname + " " + self._second_surname
        return full_name

    def populate_courses(self):
        """Choose subjects randomly and assign grades"""
        if len(self.curriculum) == 0:
            for year in self.years:
                course = []
                subjects = list(self.dataLoader.subjects)
                random.shuffle(subjects)
                subjects = subjects[0 : self.N_subjects]

                for subject in subjects:
                    subject_dict = {}
                    subject_dict["grade"] = random.randint(0, 10)
                    verbose_name = random.choice(
                        list(self.dataLoader.subjects[subject])
                    )
                    subject_dict["verbose_name"] = verbose_name
                    course.append({str(subject + year): subject_dict})

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
