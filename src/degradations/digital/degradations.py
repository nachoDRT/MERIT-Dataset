from utils import *
from io import BytesIO
from tqdm import tqdm


def generate_paragraph_samples(merit_subset_iterator, lang: str, data_format: str = "seq"):

    images_bytes = []
    ground_truths = []

    if data_format == "seq":
        d_features = read_dataset_features_json()

    for i, sample in enumerate(merit_subset_iterator):

        _, annotations = get_sample_data(sample)
        text = clean_paragraph_annotation(annotations)
        img = generate_img(text)

        buffer = BytesIO()
        img[0].save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        if data_format == "seq":
            annotations = format_annotations_cordv2_style(annotations, d_features[f"years-{lang}"])
        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}


# TODO
def generate_line_samples(merit_subset_iterator, lang: str, data_format: str = "seq"):

    images_bytes = []
    ground_truths = []

    if data_format == "seq":
        d_features = read_dataset_features_json()
    if lang == "es":
        years = ["3_de_la_eso", "4_de_la_eso", "1_de_bachillerato", "2_de_bachillerato"]
    elif lang == "en":
        years = ["year_9", "year_10", "year_11", "year_12"]

    for i, sample in tqdm(enumerate(merit_subset_iterator)):

        _, annotations = get_sample_data(sample)
        text = clean_line_annotation(annotations, years)
        img = generate_line_img(text)
        # img[0].show()

        buffer = BytesIO()
        img[0].save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        if data_format == "seq":
            annotations = format_annotations_cordv2_style(annotations, d_features[f"years-{lang}"])
        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}
