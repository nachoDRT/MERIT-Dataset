import re
import os
import cv2
import json
import torch
import utils
import logging
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from datasets import load_dataset, get_dataset_config_names
from donut import JSONParseEvaluator
from transformers import DonutProcessor, VisionEncoderDecoderModel
import argparse
from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm


SALIENCY = False
SALIENCIES_ROOT = "/app/saliencies/files/"
SAMPLES_LIMIT = 218


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def get_donut(subfolder: str):
    log_info("Loading Model and Processor")

    print(subfolder)
    model = VisionEncoderDecoderModel.from_pretrained("de-Rodrigo/donut-merit", subfolder=subfolder)

    processor = DonutProcessor.from_pretrained("de-Rodrigo/donut-merit", subfolder=subfolder)

    return model, processor


def resize_image(image, new_width):

    original_width, original_height = image.size
    new_height = int((new_width / original_width) * original_height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    dataset = load_dataset(
        dataset_name, subset_name, split="test", streaming=True
    )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def get_sample_data(sample):

    # log_info("Getting Image")

    img = sample["image"]
    # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Convert the image to RGB if it's not already (this ensures 3 color channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    gt = sample["ground_truth"]
    gt = gt.replace("'", '"')
    gt = json.loads(gt)
    # gt = gt["gt_parse"]

    if SALIENCY:
        img = resize_image(img, 512)

    return img, gt


def compute_saliency(outputs, pixels, donut_p, image):

    token_logits = torch.stack(outputs.scores, dim=1)
    token_probs = torch.softmax(token_logits, dim=-1)
    token_texts = []

    for token_index in range(len(token_probs[0])):

        target_token_prob = token_probs[
            0, token_index, outputs.sequences[0, token_index]
        ]

        if pixels.grad is not None:
            pixels.grad.zero_()

        target_token_prob.backward(retain_graph=True)

        saliency = pixels.grad.data.abs().squeeze().mean(dim=0)

        token_id = outputs.sequences[0][token_index].item()
        token_text = donut_p.tokenizer.decode([token_id])
        log_info(f"Considered sequence token: {token_text}")

        safe_token_text = re.sub(r'[<>:"/\\|?*]', "_", token_text)
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")

        unique_safe_token_text = f"{safe_token_text}_{current_datetime}"
        file_name = f"saliency_{unique_safe_token_text}.png"

        saliency = utils.convert_tensor_to_rgba_image(saliency)

        """Merge saliency image twice 1st: remove black background and fuse, 
        2nd fuse again to still see original document"""
        saliency = utils.add_transparent_image(np.array(image), saliency)
        saliency = utils.convert_rgb_to_rgba_image(saliency)
        saliency = utils.add_transparent_image(np.array(image), saliency, 0.7)

        saliency = utils.label_frame(saliency, token_text)

        save_img(saliency, os.path.join(SALIENCIES_ROOT, file_name))
        token_texts.append(token_text)

    utils.saliency_video(SALIENCIES_ROOT, token_texts)

    return token_index


def save_img(img, path):

    if img.dtype != np.uint8:
        img = (255 * img / np.max(img)).astype(np.uint8)
    cv2.imwrite(path, img)


def compute_output(donut_m, donut_p, evaluator, pixels, image, gt):
    # log_info("Computing Output")

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = donut_p.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # f***g hell, it doesn't fit on the GPU
    device = "cpu"
    donut_m.to(device)

    pixels = pixels.to(device)
    pixels.requires_grad = True

    outputs = donut_m.generate.__wrapped__(
        model,
        pixels,
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=donut_m.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=donut_p.tokenizer.pad_token_id,
        eos_token_id=donut_p.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[donut_p.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    if SALIENCY:
        compute_saliency(outputs, pixels, donut_p, image)

    sequence = donut_p.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(donut_p.tokenizer.eos_token, "").replace(
        donut_p.tokenizer.pad_token, ""
    )
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    seq = processor.token2json(sequence)
    score = evaluator.cal_acc(seq, gt)

    return seq, score


def process_dataset(dataset_iterator):
    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # Get image and ground truth
        image, gt = get_sample_data(sample)

        # Prepare image
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Get output
        output_seq, score = compute_output(
            model, processor, evaluator, pixel_values, image, gt
        )

        accs.append(score)
        output_list.append(output_seq)
        # log_info(f"Grades detected: {output_seq}")

        if i + 1 >= SAMPLES_LIMIT:
            break

    return accs, output_list


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    init_hf_hub()

    dataset_name = args.dataset
    subset_name = args.subset
    donut_model_version = args.model

    if subset_name == "all":
        subsets = get_dataset_config_names(dataset_name)
    else:
        subsets = [subset_name]

    # Project config
    logging.basicConfig(level=logging.INFO)

    # Load model and processor
    model, processor = get_donut(donut_model_version)

    for subset_name in subsets:
        print(f"Processing {subset_name}")

        # Get dataset
        dataset_iter = get_dataset_iterator(dataset_name, subset_name)

        # Process dataset
        accs, outputs_list = process_dataset(dataset_iter)
        print(f"Mean accuracy {subset_name}: {np.mean(accs)}")
        # print(outputs_list)
        print(accs)
