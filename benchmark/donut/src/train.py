import argparse
import debugpy
import json
import random
import torch
import re
import os
import wandb
from huggingface_hub import login, HfApi
import pytorch_lightning as pl
import numpy as np
from datasets import load_dataset
from datasets import Image as Image_d
from transformers import (
    VisionEncoderDecoderConfig,
    DonutProcessor,
    VisionEncoderDecoderModel,
)
from typing import Any, List, Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset, Dataset
from nltk import edit_distance
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
from PIL import Image
import ast
from typing import Union

HF_CARD_FILES = [
    "/app/src/card/README.md",
    "/app/src/card/.huggingface.yaml",
    "/app/src/card/assets/dragon_huggingface.png",
]

WANDB_PROJECT = "MERIT-Dataset-Img2Sequence"


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, freeze_encoder):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            self.model.encoder.eval()

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    # def configure_optimizers(self):
    #     # you could also add a learning rate scheduler if you want
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

    #     return optimizer

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


class DonutDataset(Dataset):
    """
    PyTorch Dataset for Donut. Loads a HuggingFace Dataset and optionally filters
    samples based on their image filename. If the subset lacks image paths, loads
    the base "es-digital-seq" to extract filenames and indices, then applies these
    indices to the actual subset (e.g., "es-digital-zoom-degradation-seq").

    Args:
        dataset_name_or_path (str): Name of the dataset on Hugging Face or local path.
        subset (str): Subset or configuration name of the dataset.
        max_length (int): Maximum number of tokens for target sequences.
        split (str): Split to load ("train", "validation", or "test").
        ignore_id (int): Ignore index value for loss computation.
        task_start_token (str): Special start token for the decoder.
        prompt_end_token (str): Special end token for the decoder.
        sort_json_key (bool): Whether to sort JSON keys in output.
        filter_substring (str or List[str], optional): Substring or list of substrings
            that the image filename must contain.
        base_subset (str): Fallback subset to load filenames from if current subset
            lacks image paths. Defaults to "es-digital-seq".
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        subset: str,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        filter_substring: Union[str, List[str]] = None,
        base_subset: str = None,
    ):
        super().__init__()

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token or task_start_token
        self.sort_json_key = sort_json_key

        # Normalize filter_substring to a list
        if filter_substring:
            self.filter_substring = (
                [filter_substring]
                if isinstance(filter_substring, str)
                else filter_substring
            )
        else:
            self.filter_substring = None

        # Determine if we need fallback for filenames
        name_indices = None
        # if self.filter_substring and subset != base_subset:
        if self.filter_substring:
            # Load base dataset to extract filenames and indices
            if base_subset:

                name_ds = load_dataset(
                    dataset_name_or_path,
                    name=base_subset,
                    split=self.split,
                    num_proc=8
                )
            else:
                name_ds = load_dataset(
                    dataset_name_or_path,
                    split=self.split,
                    num_proc=8
                )
            # Identify image column in base
            name_image_cols = [col for col, feat in name_ds.features.items()
                                if isinstance(feat, Image_d)]
            if not name_image_cols:
                raise RuntimeError(
                    f"Error: no image column in base subset '{base_subset}'. "
                    f"Available columns: {list(name_ds.features.keys())}"
                )
            name_col = name_image_cols[0]
            # Cast to get path only
            name_ds = name_ds.cast_column(name_col, Image_d(decode=False))

            # Manually record indices where any substring is in the path
            substrs = self.filter_substring
            try:
                name_indices = [
                    i for i, ex in enumerate(name_ds)
                    if any(sub in ex[name_col]["path"] for sub in substrs)
                ]
            except Exception as e:
                raise RuntimeError(
                    f"Error while extracting indices from base subset: {e}"
                )

            if not name_indices:
                raise RuntimeError(
                    f"No samples in base subset '{base_subset}' match substrings {substrs!r}."
                )

        # Load the actual subset dataset
        if dataset_name_or_path == "dvgodoy/rvl_cdip_mini":
            self.dataset = load_dataset(
                dataset_name_or_path,
                split=self.split,
                num_proc=8
            )
        elif dataset_name_or_path == "naver-clova-ix/cord-v2":
            self.dataset = load_dataset(
                dataset_name_or_path,
                split=self.split,
                num_proc=8
            )
        else:
            self.dataset = load_dataset(
                dataset_name_or_path,
                name=subset,
                split=self.split,
                num_proc=8
            )

        # If fallback indices exist, apply them
        if name_indices is not None:
            self.dataset = self.dataset.select(name_indices)

        # Now apply image filtering on the loaded dataset if needed
        if self.filter_substring:
            # Identify image column(s)
            image_cols = [col for col, feat in self.dataset.features.items()
                          if isinstance(feat, Image_d)]
            if not image_cols:
                raise RuntimeError(
                    f"Error: no image column found in subset '{subset}'. "
                    f"Available columns: {list(self.dataset.features.keys())}"
                )
            image_column = image_cols[0]

            # Cast to disable decoding for path access
            self.dataset = self.dataset.cast_column(
                image_column,
                Image_d(decode=False)
            )

            # Show sample filenames for verification
            sample_limit = min(10, len(self.dataset))
            sample_records = self.dataset.select(range(sample_limit))
            filenames = [rec[image_column]["path"] for rec in sample_records]
            print(f"First {sample_limit} filenames after fallback and filtering: {filenames}")

            # Re-cast back to decoded images for __getitem__
            self.dataset = self.dataset.cast_column(
                image_column,
                Image_d(decode=True)
            )

            print(f"Loaded {len(self.dataset)} samples after applying filter {self.filter_substring!r}.")
        else:
            print(f"Loaded {len(self.dataset)} samples.")

        self.dataset_length = len(self.dataset)
        if self.filter_substring:
            print(f"Loaded {self.dataset_length} samples after applying filter {self.filter_substring!r}.")
        else:
            print(f"Loaded {self.dataset_length} samples.")

        # Prepare target token sequences
        self.gt_token_sequences = []
        self.added_tokens = []

        if dataset_name_or_path == "de-Rodrigo/merit" or dataset_name_or_path == "naver-clova-ix/cord-v2":
            for sample in self.dataset:
                try:
                    ground_truth = json.loads(sample["ground_truth"])
                except json.decoder.JSONDecodeError:
                    ground_truth = ast.literal_eval(sample["ground_truth"])

                # Normalize ground truth to a list of JSON objects
                if "gt_parses" in ground_truth:
                    gt_jsons = ground_truth["gt_parses"]
                elif "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict):
                    gt_jsons = [ground_truth["gt_parse"]]
                elif isinstance(ground_truth, dict):
                    gt_jsons = [ground_truth]
                else:
                    raise ValueError("Unexpected ground_truth format.")

                # Convert each JSON to token sequence
                self.gt_token_sequences.append([
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=(self.split == "train"),
                        sort_json_key=self.sort_json_key,
                    ) + processor.tokenizer.eos_token
                    for gt_json in gt_jsons
                ])

        elif dataset_name_or_path == "dvgodoy/rvl_cdip_mini":
            for sample in self.dataset:

                ocr_words = sample["ocr_words"]

                words_list = [{"word": word} for word in ocr_words]
                page = {"page_0": words_list}
                ground_truth = {"gt_parse": page}

                # Normalize ground truth to a list of JSON objects
                if "gt_parses" in ground_truth:
                    gt_jsons = ground_truth["gt_parses"]
                elif "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict):
                    gt_jsons = [ground_truth["gt_parse"]]
                elif isinstance(ground_truth, dict):
                    gt_jsons = [ground_truth]
                else:
                    raise ValueError("Unexpected ground_truth format.")

                # Convert each JSON to token sequence
                self.gt_token_sequences.append([
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=(self.split == "train"),
                        sort_json_key=self.sort_json_key,
                    ) + processor.tokenizer.eos_token
                    for gt_json in gt_jsons
                ])

        # Add special tokens to tokenizer
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads the image from the dataset at the given index, processes it into a tensor,
        and tokenizes the corresponding ground truth sequence.

        Returns:
            pixel_values: Preprocessed image tensor.
            labels: Tokenized ground truth sequence with masked tokens (model doesn't need to predict prompt or pad tokens).
            target_sequence: The original ground truth string.
        """
        # Retrieve the sample at index 'idx' from the dataset
        sample = self.dataset[idx]

        # Load the image from the sample
        image = sample["image"]

        # If the image is not a PIL Image, try converting it (e.g., from a NumPy array)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Convert the image to RGB if it's not already (this ensures 3 color channels)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process the image using the processor
        # 'random_padding' is enabled for training to introduce variability
        pixel_values = processor(image, random_padding=self.split == "train", return_tensors="pt").pixel_values
        # Remove any extra dimensions added by the processor
        pixel_values = pixel_values.squeeze()

        # Randomly select one target sequence from the available ground truth sequences for this sample
        target_sequence = random.choice(self.gt_token_sequences[idx])

        # Tokenize the target sequence without adding special tokens
        # The sequence is padded or truncated to 'max_length' and converted into a tensor
        input_ids = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Clone the tokenized input_ids to create labels, and mask out the pad tokens
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id

        # Return the processed image tensor, the labels, and the original target sequence
        return pixel_values, labels, target_sequence


class PushToHubCallback(Callback):
    """
    Callback to push the model to the Hugging Face Hub only when
    the monitored validation metric improves.
    """
    def __init__(
        self,
        model_output_name: str,
        dataset_subset: str,
        monitor: str = "val_edit_distance",
        mode: str = "min",
        save_dir: str = "checkpoints",
    ):
        super().__init__()
        self.api = HfApi()
        self.model_output_name = model_output_name
        self.dataset_subset = dataset_subset
        self.save_dir = save_dir
        self.monitor = monitor

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        # Initialize best_score according to mode
        self.best_score = float('inf') if mode == "min" else -float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch. If the monitored metric improves,
        save and push the model.
        """
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        has_improved = (
            current_score < self.best_score if self.mode == "min" else current_score > self.best_score
        )
        if has_improved:
            print(f"Detected improvement in {self.monitor}: {self.best_score} -> {current_score}")
            self.best_score = current_score
            self._push_model(trainer, pl_module, epoch=trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        """
        Called once training is complete. Only push the card files (README, configs, etc.)
        to the Hub.
        """
        print("Uploading final card files to the Hub...")
        repo_id = f"de-Rodrigo/{self.model_output_name}"
        self._upload_card_files(repo_id)

    def _push_model(self, trainer, pl_module, epoch: int):
        """
        Save the model and processor locally and push them to the Hub.
        """
        save_path = os.path.join(
            self.save_dir,
            f"{self.model_output_name}_{self.dataset_subset}_epoch{epoch}",
        )
        pl_module.model.save_pretrained(save_path)
        pl_module.processor.save_pretrained(save_path)

        repo_id = f"de-Rodrigo/{self.model_output_name}"
        self.api.upload_folder(
            folder_path=save_path,
            path_in_repo=f"{self.dataset_subset}_MERIT-paper",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Best model up to epoch {epoch} ({self.monitor}={self.best_score})",
        )
        # Upload additional files as well
        self._upload_card_files(repo_id)

    def _upload_card_files(self, repo_id: str):
        """
        Upload additional card files (README, configs, etc.) to the repository.
        """
        for file_path in HF_CARD_FILES:
            print(f"Uploading {file_path} to {repo_id}")
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo="/".join(file_path.split(os.sep)[4:]),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Uploading card files",
            )


def load_session_datasets(dataset_name: str, subset: str = "", school_name_subsets: str = None):

    train_dataset = DonutDataset(
        dataset_name,
        subset=subset,
        max_length=max_length,
        split="train",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
        filter_substring=school_name_subsets,
        base_subset=base_subset
    )

    val_dataset = DonutDataset(
        dataset_name,
        subset=subset,
        max_length=max_length,
        split="validation",
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
        filter_substring=school_name_subsets,
        base_subset=base_subset
    )

    return train_dataset, val_dataset


def load_session_dataset_train(dataset_name: str, subset: str = "", school_name_subsets: str = None, split: str = "train"): 
    
    train_dataset = DonutDataset(
        dataset_name,
        subset=subset,
        max_length=max_length,
        split=split,
        task_start_token="<s_cord-v2>",
        prompt_end_token="<s_cord-v2>",
        sort_json_key=False,
        filter_substring=school_name_subsets
    )

    return train_dataset


def load_secret_dataset(dataset_name: str, subsets: list):

    train_datasets = []
    val_datasets = []

    for subset in subsets:
        print(subset)
        train_ds = DonutDataset(
            dataset_name,
            subset=subset,
            max_length=max_length,
            split="test",
            task_start_token="<s_cord-v2>",
            prompt_end_token="<s_cord-v2>",
            sort_json_key=False,
        )
        val_ds = DonutDataset(
            dataset_name,
            subset=subset,
            max_length=max_length,
            split="test",
            task_start_token="<s_cord-v2>",
            prompt_end_token="<s_cord-v2>",
            sort_json_key=False,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    return train_dataset, val_dataset


def load_secret_dataset_as_validation(dataset_name: str, subsets: list):
    
    val_datasets = []
    
    for subset in subsets:
        val_ds = DonutDataset(
            dataset_name,
            subset=subset,
            max_length=max_length,
            split="test",
            task_start_token="<s_cord-v2>",
            prompt_end_token="<s_cord-v2>",
            sort_json_key=False,
        )
        val_datasets.append(val_ds)

    val_dataset = ConcatDataset(val_datasets)
    
    return val_dataset


def get_school_combination_name(comb: tuple) -> str:
    name = ""

    for element in comb:
        if name != "":
            name += "-"
        name += element

    return name


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--base_subset", type=str)
    parser.add_argument("--dataset_subsets", type=str, action='append')
    parser.add_argument("--school_name_subsets", type=str, action='append', default=None)
    parser.add_argument("--test_real", action="store_true", default=False)
    parser.add_argument("--freeze_encoder", action="store_true", default=False)
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    # Define constants
    freeze_encoder = args.freeze_encoder
    dataset = args.dataset_name
    base_subset = args.base_subset
    dataset_subsets = args.dataset_subsets
    school_name_subsets = args.school_name_subsets
    test_real = args.test_real
    model_output_name = "".join(["donut-", dataset.split('/')[-1]])

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    # Load model and processor
    image_size = [1280, 960]
    max_length = 768

    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

    # Dataset instances
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False

    if dataset == "de-Rodrigo/merit-secret":
        subsets_disponibles = [
            "britanico",
            "fomento",
            "maravillas",
            "mater",
            "montealto",
            "pilar",
            "recuerdo",
            "retamar",
            "sanpablo",
            "sanpatricio",
        ]
        train_dataset, val_dataset = load_secret_dataset(dataset, subsets_disponibles)

    elif dataset == "de-Rodrigo/merit-aux":
        
        train_dataset = []

        train_ds_table = load_session_dataset_train(dataset_name=dataset, subset=dataset_subsets[0])
        train_dataset.append(train_ds_table)

        train_ds_alpha = load_session_dataset_train(dataset_name="de-Rodrigo/merit", subset="es-digital-seq", school_name_subsets="britanico", split="test")
        train_dataset.append(train_ds_alpha)

        train_dataset = ConcatDataset(train_dataset)

        if test_real:
            dataset = "de-Rodrigo/merit-secret"
            subsets_disponibles = ['britanico', 'fomento', 'maravillas', 'mater', 'montealto', 'pilar', 'recuerdo', 'retamar', 'sanpablo', 'sanpatricio']
            val_dataset = load_secret_dataset_as_validation(dataset, subsets_disponibles)

    elif dataset == "combination":

        train_dataset = []
        val_dataset = []

        dataset_subsets = ["es-render-seq"]

        for subset in dataset_subsets:
            train_ds, val_ds = load_session_datasets(dataset_name="de-Rodrigo/merit", subset=subset, school_name_subsets=school_name_subsets)
            train_dataset.append(train_ds)
            # val_dataset.append(val_ds)

        train_ds_table = load_session_dataset_train(dataset_name="de-Rodrigo/merit-aux", subset="retamar_train-asc-synth")
        train_dataset.append(train_ds_table)

        train_ds_alpha = load_session_dataset_train(dataset_name="de-Rodrigo/merit", subset="es-digital-seq", school_name_subsets="britanico", split="test")
        train_dataset.append(train_ds_alpha)

        train_dataset = ConcatDataset(train_dataset)
        # val_dataset = ConcatDataset(val_dataset)

        if test_real:
            dataset = "de-Rodrigo/merit-secret"
            subsets_disponibles = ['britanico', 'fomento', 'maravillas', 'mater', 'montealto', 'pilar', 'recuerdo', 'retamar', 'sanpablo', 'sanpatricio']
            val_dataset = load_secret_dataset_as_validation(dataset, subsets_disponibles)

    else:

        train_dataset = []
        val_dataset = []

        for subset in dataset_subsets:
            train_ds, val_ds = load_session_datasets(dataset_name=dataset, subset=subset, school_name_subsets=school_name_subsets)
            train_dataset.append(train_ds)
            val_dataset.append(val_ds)

        train_dataset = ConcatDataset(train_dataset)
        val_dataset = ConcatDataset(val_dataset)

        if test_real:
            dataset = "de-Rodrigo/merit-secret"
            subsets_disponibles = ['britanico', 'fomento', 'maravillas', 'mater', 'montealto', 'pilar', 'recuerdo', 'retamar', 'sanpablo', 'sanpatricio']
            val_dataset = load_secret_dataset_as_validation(dataset, subsets_disponibles)

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_cord-v2>"])[0]

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    if test_real:
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
    else:
        percentage = 0.1
        num_samples = int(len(val_dataset) * percentage)
        indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)
        val_dataloader = DataLoader(val_dataset, batch_size=1, sampler=sampler, num_workers=4)

    # Train
    val_check_period = 0.05

    config = {
        "max_steps": 10000,
        "val_check_interval": val_check_period,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "num_training_samples_per_epoch": 800,
        "lr": 3e-5,
        "train_batch_sizes": [8],
        "val_batch_sizes": [1],
        "num_nodes": 1,
        "warmup_steps": 300,
        "result_path": "./result",
        "verbose": True,
    }

    model_module = DonutModelPLModule(config, processor, model, freeze_encoder)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    session_name = "_".join(dataset_subsets)
    # if school_name_subsets:
    #     shool_combination_name = get_school_combination_name(school_name_subsets)
    #     session_name = "_".join(dataset_subsets) + "_filtered_" + shool_combination_name
    
    # if freeze_encoder:
    #     session_name += "-frozen-encoder"

    # if dataset == "combination":
    #     session_name += "-combination"

    session_name = f"donut_{session_name}"
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=session_name, entity="iderodrigo")

    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=int(1/val_check_period)*5, verbose=False, mode="min")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.get("max_epochs"),
        max_steps=config.get("max_steps"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[early_stop_callback, PushToHubCallback("donut-merit", session_name)],
    )

    trainer.fit(model_module)
