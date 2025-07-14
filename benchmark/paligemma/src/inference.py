from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import HfApi, HfFolder
import os
import logging
from datasets import load_dataset
import argparse
from donut import JSONParseEvaluator
import numpy as np
from tqdm import tqdm
import re
import json
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from peft import PeftModel, prepare_model_for_kbit_training
from PIL import Image


WANDB_PROJECT = "MERIT-Dataset-Img2Sequence"
FINETUNED_MODEL_ID = "nielsr/paligemma-cord-demo"
# REPO_ID = "google/paligemma-3b-pt-224"
MAX_LENGTH = 512
PROMPT = "extract JSON."
LIMIT = 218


def log_info(msg: str):
    print("")
    logging.info(msg)
    print("")


def init_hf_hub():
    HfFolder.save_token(os.environ["HUGGINGFACE_HUB_TOKEN"])


def init_wandb():
    session_name = f"paligemma_test_{subset_name}"
    # Inicia un run y un logger de Lightning
    run = wandb.init(
        project=WANDB_PROJECT,
        name=session_name,
        entity="iderodrigo",
        reinit=True
    )
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=session_name,
        entity="iderodrigo",
        log_model=False
    )
    return run, wandb_logger


def get_paligemma(paligemma_model_version: str, subfolder: str):
    log_info("Loading Model and Processor")

    # Configuración de cuantización 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 1) Carga del modelo base cuantizado
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # 2) Inyección de los pesos LoRA entrenados
    model = PeftModel.from_pretrained(
        base_model,
        paligemma_model_version,
        subfolder=subfolder,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()

    # Carga del processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(
        paligemma_model_version,
        subfolder=subfolder,
    )

    return model, processor


def get_dataset_iterator(dataset_name: str, subset_name: str):
    log_info("Loading Dataset")

    if dataset_name == "de-Rodrigo/merit":
        dataset = load_dataset(
            dataset_name, subset_name, split="test", streaming=True
        )
    else:
        dataset = load_dataset(
            dataset_name, split="test", streaming=True
        )
    dataset_iterator = iter(dataset)

    return dataset_iterator


def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def get_sample_data(sample):

    img = sample["image"]

    # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    # Convert the image to RGB if it's not already (this ensures 3 color channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    if dataset_name == "de-Rodrigo/merit" or dataset_name == "naver-clova-ix/cord-v2":
        gt = sample["ground_truth"]
        if dataset_name == "de-Rodrigo/merit":
            gt = gt.replace("'", '"')
        gt = json.loads(gt)

        if dataset_name == "naver-clova-ix/cord-v2":
            gt = gt["gt_parse"]
        
    else:
        ocr_words = sample["ocr_words"]
        words_list = [{"word": word} for word in ocr_words]
        page = {"page_0": words_list}
        gt = {"gt_parse": page}

    print(gt)


    return img, gt


def process_dataset(dataset_iterator):
    evaluator = JSONParseEvaluator()
    accs = []
    output_list = []

    for i, sample in tqdm(enumerate(dataset_iterator)):

        # Get image and ground truth
        image, gt = get_sample_data(sample)

        inputs = processor(text=PROMPT, images=image, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Autoregressively generate
        # We use greedy decoding here, for more fancy methods see https://huggingface.co/blog/how-to-generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)

            # Next we turn each predicted token ID back into a string using the decode method
            # We chop of the prompt, which consists of image tokens and our text prompt
            image_token_index = model.config.image_token_index
            num_image_tokens = len(generated_ids[generated_ids==image_token_index])
            num_text_tokens = len(processor.tokenizer.encode(PROMPT))
            num_prompt_tokens = num_image_tokens + num_text_tokens + 2
            generated_text = processor.batch_decode(generated_ids[:, num_prompt_tokens:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_json = token2json(generated_text)
            print("PREDICTION", generated_json)

            score = evaluator.cal_acc(generated_json, gt)

            accs.append(score)
            output_list.append(generated_json)

            if i >= LIMIT:
                break

    return accs, output_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--subset_name", type=str)
    parser.add_argument("--paligemma_model_version", type=str)
    parser.add_argument("--subfolder", type=str)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    subset_name = args.subset_name
    paligemma_model_version = args.paligemma_model_version
    subfolder = args.subfolder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_hf_hub()
    run, wandb_logger = init_wandb()

    model, processor = get_paligemma(paligemma_model_version, subfolder)
    dataset_iter = get_dataset_iterator(dataset_name, subset_name)

    # Process dataset
    accs, outputs_list = process_dataset(dataset_iter)

    f1 = np.mean(accs)
    print(f"Mean accuracy {subset_name}: {f1}")
    print(accs)

    wandb.log({"test_f1": f1})
    run.finish()