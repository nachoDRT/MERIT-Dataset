from utils import *
from io import BytesIO
from tqdm import tqdm
import random
from PIL import Image
import numpy as np


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


def generate_rotation_samples(merit_subset_iterator, data_format: str = "seq"):

    images_bytes = []
    ground_truths = []

    for i, sample in tqdm(enumerate(merit_subset_iterator)):

        img, annotations = get_sample_data(sample)

        angle = random.random() * 30
        if random.choice([True, False]):
            rot_angle = angle
        else:
            rot_angle = -angle
        rotated_img = img.rotate(rot_angle, expand=True, fillcolor=(0, 0, 0))

        buffer = BytesIO()
        rotated_img.save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}


def generate_zoom_samples(merit_subset_iterator, scale: float = None):
    images_bytes = []
    ground_truths = []

    for i, sample in tqdm(enumerate(merit_subset_iterator)):

        img, annotations = get_sample_data(sample)

        scaled_img = scale_img(img, scale=scale)

        buffer = BytesIO()
        scaled_img.save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}


def generate_rotation_zoom_samples(merit_subset_iterator):

    images_bytes = []
    ground_truths = []

    for i, sample in tqdm(enumerate(merit_subset_iterator)):

        img, annotations = get_sample_data(sample)

        angle = random.random() * 30
        if random.choice([True, False]):
            rot_angle = angle
        else:
            rot_angle = -angle
        rotated_img = img.rotate(rot_angle, expand=True, fillcolor=(0, 0, 0))
        scaled_img = scale_img(rotated_img)

        buffer = BytesIO()
        scaled_img.save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}


def scale_img(img, scale: float = None):

    if not scale:
        scale = random.uniform(1, 0.3)

    width, height = img.size
    center_x, center_y = width / 2, height / 2

    inv_scale = 1 / scale
    a = inv_scale  # x scale
    b = 0  # No rotation
    c = center_x - center_x * inv_scale  # Translation to center x
    d = 0  # No rotation
    e = inv_scale  # y scale
    f = center_y - center_y * inv_scale  # Translation to center y

    matrix = (a, b, c, d, e, f)

    scaled_img = img.transform(img.size, Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    return scaled_img


def generate_noisy_samples(merit_subset_iterator, seed: int | None = 42):

    images_bytes = []
    ground_truths = []
    snr_metric = []

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for i, sample in tqdm(enumerate(merit_subset_iterator)):

        img, annotations = get_sample_data(sample)
        amount = np.random.uniform(0.1, 0.45)
        noisy_img = add_noise(img, amount=amount)
        # noisy_img.show()

        buffer = BytesIO()
        noisy_img.save(buffer, format="PNG")
        images_bytes.append(buffer.getvalue())

        snr_metric.append(snr(img, noisy_img))

        ground_truths.append(json.dumps(annotations))

    return {"image": images_bytes, "ground_truth": ground_truths}, snr_metric


def add_noise(
    img: Image.Image,
    mode: str = "salt_pepper",
    amount: float = 0.02,
    mean: float = 0.0,
    std: float = 0.02,
) -> Image.Image:
    """
    Get a noisy sample.

    mode      : 'salt_pepper', 'gaussian' o 'speckle'
    amount    : fraction of altered pixels (salt & pepper)
    mean, std : noise parameters gaussian / speckle
    """

    arr = np.array(img).astype(np.float32)

    if mode == "salt_pepper":

        mask = np.random.choice([0, 1, 2], size=arr.shape[:2], p=[1 - amount, amount / 2, amount / 2])
        salt = mask == 1
        pepper = mask == 2

        arr[salt] = 255
        arr[pepper] = 0

    elif mode == "gaussian":
        noise = np.random.normal(mean, std * 255, arr.shape)
        arr += noise

    elif mode == "speckle":
        noise = np.random.normal(mean, std, arr.shape)
        arr += arr * noise

    else:
        raise ValueError(f"Non supported noise mode: {mode}")

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def snr(original: Image.Image, noisy: Image.Image) -> float:
    """
    Compute SNR between original and noisy image.
    """

    orig = np.asarray(original).astype(np.float32)
    noisy_arr = np.asarray(noisy).astype(np.float32)
    noise = orig - noisy_arr

    signal_power = np.sum(orig**2)
    noise_power = np.sum(noise**2) + 1e-8

    return signal_power / noise_power
