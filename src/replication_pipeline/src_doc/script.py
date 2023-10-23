import records_creator
import annotations_creator

# import azure_ocr
from config import *
import tqdm


from importlib import reload

# reload(azure_ocr)
reload(records_creator)
reload(annotations_creator)

print("Local modules reimported correctly")
print(
    "Starting at student {}, for a total of {} students".format(FIRST_STUDENT, STUDENTS)
)

# Create records for each student

delete_files(FIRST_STUDENT)  # Deletes files from FIRST_STUDENT onwards

for student_n in tqdm.trange(
    FIRST_STUDENT, STUDENTS, initial=FIRST_STUDENT, total=STUDENTS
):
    # Create fictional student records
    record = records_creator.SchoolRecord(student_n, db_path)

    # Create pdf and png documents
    pdf_file_path = record.create_pdf()
    png_paths = record.create_pngs()

    # Create annotations
    # Annotations are not created correctly yet
    annotations_class = annotations_creator.AnnotationsCreator(pdf_file_path)
    annotations_class.create_annotations(pdf_file_path)

    # Tag subjects and grades
    annotations_class.update_subject_grades_tags(
        record.student.curriculum, courses_pages_array
    )

    # Tag courses
    annotations_class.update_annotations_tags_courses()
    annotations_class.sort_annotations()

    # Write to json
    annotations_class.dump_annotations_json(png_paths)

    # Optionally output annotations view of pdf
    # annotations_view = annotations_creator.AnnotationsView(annotations_class.annotations, pdf_file_path)
    # annotations_view.output_annotations_view()
    # annotations_view.output_annotations_words_view()

    # Next time start from next student
    FIRST_STUDENT = student_n
