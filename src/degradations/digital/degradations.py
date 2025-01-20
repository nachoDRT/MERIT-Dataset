from typing import List
import os
from utils import *


def generate_paragraph_samples(merit_subset_paths: List):

    for path in merit_subset_paths:
        print(path)
        sample_name = os.path.splitext(os.path.basename(path))[0]
        annotation = get_annotation(path)
        text = clean_annotation(annotation)
        img = generate_img(text)
        saving_path = join(dirname(dirname(path)), "degradations", "paragraph", f"{sample_name}.png")
        print(saving_path)
        save_sample(saving_path, img)
        break
