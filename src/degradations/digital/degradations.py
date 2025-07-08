from utils import *
from io import BytesIO
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import gc
from datasets import Dataset, Value, Features, concatenate_datasets, Image as HFImage
from os.path import dirname, join, abspath
from PIL import Image as PILImage


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

    elif data_format == "token-class":

        for i, sample in enumerate(merit_subset_iterator):

            _, annotations = get_sample_data(sample)
            # text = clean_line_annotation(annotations, years)
            text, form_annotation = clean_line_annotation_token_class(annotations)
            img, transformed_annotation = generate_line_img_token_class(text, form_annotation)

            # img[0].show()
            # print(transformed_annotation)

            buffer = BytesIO()
            img[0].save(buffer, format="PNG")
            images_bytes.append(buffer.getvalue())

            ground_truths.append(json.dumps(transformed_annotation))

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


def generate_noisy_samples_stream(iterator, *, batch_size=512, seed=42):
    """
    (split Dataset, list_snr_ratio).
    Batch processing -> Efficient RAM usage.
    """
    random.seed(seed)
    np.random.seed(seed)

    feats = Features(
        {
            "image": HFImage(),
            "ground_truth": Value("string"),
            # "snr_ratio": Value("float32"),
        }
    )
    batch = {k: [] for k in feats}
    writer = None
    snr_all = []

    for idx, sample in tqdm(enumerate(iterator), desc="adding noise"):
        img, ann = get_sample_data(sample)
        noisy = add_noise(img, amount=np.random.uniform(0.1, 0.45))

        buf = BytesIO()
        noisy.save(buf, format="PNG")

        ratio = snr_ratio(img, noisy)
        snr_all.append(ratio)

        batch["image"].append(buf.getvalue())
        batch["ground_truth"].append(json.dumps(ann))
        # batch["snr_ratio"].append(ratio)

        # Free buffers every 100 imgs
        del img, noisy, buf
        if (idx + 1) % 100 == 0:
            gc.collect()

        # Free the batch when it is full and reset
        if len(batch["image"]) >= batch_size:
            writer = _flush_batch(batch, feats, writer)
            batch = {k: [] for k in feats}

    # Last batch
    if batch["image"]:
        writer = _flush_batch(batch, feats, writer)

    return writer, snr_all


def generate_watermark_samples_stream(iterator, language: str, *, batch_size: int = 512, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    feats = Features(
        {
            "image": HFImage(),
            "ground_truth": Value("string"),
        }
    )

    batch = {k: [] for k in feats}
    writer = None

    decoder = HFImage()
    watermarks = load_watermarks(language)

    for idx, sample in tqdm(enumerate(iterator), desc="adding watermark"):
        img_name: str = sample["image"]["path"]
        school = img_name.split("_")[1]

        watermark_img = watermarks[school]

        decoded_img = decoder.decode_example(sample["image"])
        np_img = np.array(decoded_img)
        decoded_cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        output_img = add_transparent_image(decoded_cv_img, watermark_img, 0.2)

        _, annotations = get_sample_data(sample)

        buf = BytesIO()
        output_img.save(buf, format="PNG")

        batch["image"].append(buf.getvalue())
        batch["ground_truth"].append(json.dumps(annotations))

        if (idx + 1) % 100 == 0:
            del decoded_img, np_img, decoded_cv_img, output_img, buf
            gc.collect()

        if len(batch["image"]) >= batch_size:
            writer = _flush_batch(batch, feats, writer)
            batch = {k: [] for k in feats}

    if batch["image"]:
        writer = _flush_batch(batch, feats, writer)

    return writer


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


def snr_ratio(original: Image.Image, noisy: Image.Image) -> float:
    """Compute SNR."""
    orig = np.asarray(original).astype(np.float32)
    noise = orig - np.asarray(noisy).astype(np.float32)

    return np.sum(orig**2) / (np.sum(noise**2) + 1e-8)


def _flush_batch(batch, feats, writer):
    """Transform the dictionary batch→Dataset and concatenate with the writer."""
    ds_batch = Dataset.from_dict(batch, features=feats)
    return ds_batch if writer is None else concatenate_datasets([writer, ds_batch])


def generate_id_picture_stream(iterator, *, batch_size=512, seed=42):

    doc2IdIimg_csv_path = join(dirname(dirname(abspath(__file__))), "config", "doc_2_id_img.csv")
    print(doc2IdIimg_csv_path)

    if os.path.isfile(doc2IdIimg_csv_path):
        print("Done")

    else:
        compute_doc2IdImg_csv(doc2IdIimg_csv_path)
