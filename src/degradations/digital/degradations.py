from utils import *
from io import BytesIO


MAX_SAMPLES = 2


def generate_paragraph_samples(merit_subset_iterator, lang: str, data_format: str = "cord-v2"):

    images_bytes = []
    ground_truths = []

    if data_format == "cord-v2":
        d_features = read_dataset_features_json()

    for i, sample in enumerate(merit_subset_iterator):

        _, annotations = get_sample_data(sample)
        text = clean_annotation(annotations)
        img = generate_img(text)

        buffer = BytesIO()
        img[0].save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        if data_format == "cord-v2":
            annotations = format_annotations_cordv2_style(annotations, d_features[f"years-{lang}"])
        ground_truths.append(json.dumps(annotations))

        if i >= MAX_SAMPLES:
            break

    return {"image": images_bytes, "ground_truth": ground_truths}
