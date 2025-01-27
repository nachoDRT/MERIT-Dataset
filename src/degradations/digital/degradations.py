from utils import *
from io import BytesIO

MAX_SAMPLES = 2


def generate_paragraph_samples(merit_subset_iterator):

    images_bytes = []
    ground_truths = []

    for i, sample in enumerate(merit_subset_iterator):

        _, annotations = get_sample_data(sample)
        text = clean_annotation(annotations)
        img = generate_img(text)

        buffer = BytesIO()
        img[0].save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())
        ground_truths.append(json.dumps(annotations))

        if i >= MAX_SAMPLES:
            break

    return {"image": images_bytes, "ground_truth": ground_truths}
