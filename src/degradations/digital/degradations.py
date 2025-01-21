from utils import *


def generate_paragraph_samples(merit_subset_iterator):

    for sample in merit_subset_iterator:

        _, annotation = get_sample_data(sample)
        text = clean_annotation(annotation)
        img = generate_img(text)
        img[0].show()
        break
