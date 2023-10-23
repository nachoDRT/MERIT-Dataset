import os
import glob

# Script configuration
SCHOOL_NICKNAME = "default"
DOCS_Z_FILL = 4  # Rellena con 0s la nomenclatura
STUDENTS = 10  # Número de alumnos
FIRST_STUDENT = 0  # Primer alumno a crear
verbose_level = 0  # Level 0 = only grade, Level 1 = (sobresaliente - 9), Level 2 = SB 9

# School specific configuration
courses_pages_array = [[(0, 11)], [(1, 10)], [(2, 11)]]
# Page 0 has courses 0 with 12 subjects and 1 with 11 subjects;
# Page 1 has course 2 with 11 subjects

courses_names = ["3_de_la_eso", "4_de_la_eso", "1_de_bachillerato"]
possible_courses_global = ["Tercer curso", "Cuarto curso", "BACHILLERATO"]

# For each template the information above may need to be redefined

# Paths
parent_dir = os.path.dirname(os.getcwd())
if parent_dir == "/Users/ascuadrado/Documents/GitHub":
    parent_dir = os.path.join(parent_dir, "records-dataset")
base_path = os.path.join(parent_dir, SCHOOL_NICKNAME)
db_path = os.path.join(base_path, "database_output")

# Database paths
annotations_path = os.path.join(db_path, "annotations")
images_path = os.path.join(db_path, "images")
pdf_path = os.path.join(db_path, "synthetic_pdf_docs")
res_path = os.path.join(parent_dir, "res")
template_path = os.path.join(base_path, "template.docx")

# Auxiliary functions


def delete_files(first_file: int):
    i = first_file
    while True:
        glob_expr = db_path + "/*/*{}*".format(str(i).zfill(DOCS_Z_FILL))
        files = glob.glob(glob_expr)

        # Delete files
        i = i + 1

        if len(files) == 0:
            break
        print("Deleting files: {}".format(i))
        for file in files:
            os.remove(file)
