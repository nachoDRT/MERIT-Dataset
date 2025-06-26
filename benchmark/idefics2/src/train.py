import argparse
import debugpy
import torch
import random
import json
import re
import os
import wandb
import lightning as L
import numpy as np
from huggingface_hub import login, HfApi
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
from typing import Any, List, Dict
from nltk import edit_distance
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


"""
Script extracted from Niels Rogge GitHub with minor modifications
You can find Niels project here: 
https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2
"""


USE_LORA = False
USE_QLORA = True
USE_ADD_ADAPTER = True
MAX_LENGTH = 768
MODEL_REPO_ID = "de-Rodrigo/idefics2-merit"
HF_CARD_FILES = ["/app/src/card/README.md", "/app/src/card/.huggingface.yaml", "/app/src/card/assets/dragon_huggingface.png"]
WANDB_PROJECT = "MERIT-Dataset-Img2Sequence"
LOGGING_STEPS = 1


class Idefics2Dataset(Dataset):
    """
    PyTorch Dataset for Idefics2. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        processor,
        model,
        dataset_name_or_path: str,
        subset: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        self.processor = processor
        self.model = model

        if dataset_name_or_path == "de-Rodrigo/merit":
            self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split)
            self.dataset_length = len(self.dataset)

            self.gt_token_sequences = []
            self.added_tokens = []
            for sample in self.dataset:
                ground_truth = json.loads(sample["ground_truth"])
                if "gt_parses" in ground_truth:
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            update_special_tokens_for_json_key=self.split == "train",
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons
                    ]
                )

        elif dataset_name_or_path == "dvgodoy/rvl_cdip_mini":
            self.dataset = load_dataset(dataset_name_or_path, split=self.split)
            self.dataset_length = len(self.dataset)

            self.gt_token_sequences = []
            self.added_tokens = []
            for sample in self.dataset:
                ocr_words = sample["ocr_words"]

                words_list = [{"word": word} for word in ocr_words]
                page = {"page_0": words_list}
                ground_truth = {"gt_parse": page}

                if "gt_parses" in ground_truth:
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            update_special_tokens_for_json_key=self.split == "train",
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons
                    ]
                )

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
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
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # Inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])

        return image, target_sequence


class Idefics2ModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, labels = batch

        outputs = self.model(input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, pixel_attention_mask, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
            max_new_tokens=768)
        # turn them back into text, chopping of the prompt
        # important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=lambda examples: train_collate_fn(examples, self.processor, self.model), batch_size=self.batch_size, shuffle=True, num_workers=4)

    # def val_dataloader(self):
    #     return DataLoader(val_dataset, collate_fn=lambda examples: eval_collate_fn(examples, self.processor, self.model), batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        percentage = 0.05
        num_samples = int(len(val_dataset) * percentage)
        indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)

        return DataLoader(
            val_dataset, 
            sampler=sampler,
            collate_fn=lambda examples: eval_collate_fn(examples, self.processor, self.model),
            batch_size=self.batch_size, 
            num_workers=4
        )


class PushToHubCallback(Callback):
    """
    Callback para subir el modelo a Hugging Face Hub solamente cuando mejora la métrica de validación.
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
        self.model_output_name = f"{self.model_output_name}"
        self.dataset_subset = dataset_subset
        self.save_dir = save_dir
        self.monitor = monitor

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best_score = float('inf') if mode == "min" else -float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Subida condicional del modelo si mejora la métrica monitorizada.
        """
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)

        if current_score is None:
            return  # No hay métrica aún

        improved = (
            current_score < self.best_score if self.mode == "min"
            else current_score > self.best_score
        )

        if improved:
            print(f"[PushToHubCallback] {self.monitor} mejoró: {self.best_score} -> {current_score}")
            self.best_score = current_score
            self._push_model(trainer, pl_module)

    def _push_model(self, trainer, pl_module):
        """
        Guarda y sube el modelo a Hugging Face Hub.
        """
        epoch = trainer.current_epoch
        save_path = os.path.join(
            self.save_dir,
            f"{self.model_output_name}_{self.dataset_subset}_epoch{epoch}",
        )

        pl_module.model.save_pretrained(save_path)
        pl_module.processor.save_pretrained(save_path)

        repo_id = f"{self.model_output_name}"
        self.api.upload_folder(
            folder_path=save_path,
            path_in_repo=f"{self.dataset_subset}_MERIT-paper",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Mejor modelo hasta epoch {epoch} ({self.monitor}={self.best_score})",
        )

        self._upload_card_files(repo_id)

    def _upload_card_files(self, repo_id: str):
        """
        Sube archivos adicionales como README, config, etc.
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

def save_and_push_model(processor, model, repo_id, dataset_subset, save_dir, commit_message):

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "initial_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Guardar modelo y procesador
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # Subir al hub
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    api = HfApi()
    api.upload_folder(
        folder_path=checkpoint_dir,
        path_in_repo=dataset_subset,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )

    for file in HF_CARD_FILES:
        print(f"Uploading {file} to {repo_id}")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo='/'.join(file.split('/')[4:]),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Uploading additional files"
        )


def load_model() -> Idefics2ForConditionalGeneration:
    """ Three options for training, from the lowest precision training to the highest 
    precision training:
        - QLora
        - Standard Lora
        - Full fine-tuning
    """
    
    if USE_QLORA or USE_LORA:
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        loadel_model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            quantization_config=bnb_config if USE_QLORA else None,
        )
        if USE_ADD_ADAPTER:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
                use_dora=False if USE_QLORA else True,
                init_lora_weights="gaussian",
            )
            loadel_model.add_adapter(lora_config)
            loadel_model.enable_adapters()
    else:
        """For full fine-tuning, we can speed up the model using Flash Attention, only available
        on certain devices, see: https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features"""
        loadel_model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )

    return loadel_model


def apply_peft():
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian",
    )

    peft_model = prepare_model_for_kbit_training(peft_model)
    peft_model = get_peft_model(peft_model, lora_config)

    return peft_model


def train_collate_fn(examples, processor, model):
    texts = []
    images = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ground_truth}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == model.config.image_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, labels


def eval_collate_fn(examples, processor, model):
    images = []
    texts = []
    answers = []
    for example in examples:
        image, ground_truth = example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images.append([image])
        texts.append(text.strip())
        answers.append(ground_truth)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    pixel_attention_mask = batch["pixel_attention_mask"]

    return input_ids, attention_mask, pixel_values, pixel_attention_mask, answers


def init_pl_module(processor, model):
    configuration = {"max_epochs": 10,
        "val_check_interval": 0.05,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        "precision": "16-mixed",
        # "seed":2022,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
        "logging_steps": LOGGING_STEPS,
    }

    model_module = Idefics2ModelPLModule(configuration, processor, model)

    return model_module, configuration


def train_idefics2(idefics2, configuration):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    # wandb_logger = WandbLogger(project="Idefics2", name=args.subset)
    session_name = "_".join([args.subset])
    session_name = f"idefics2_{session_name}"
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=session_name, entity="iderodrigo")

    trainer = L.Trainer(
            accelerator="gpu",
            devices="auto", # Use available GPUs
            max_epochs=configuration.get("max_epochs"),
            check_val_every_n_epoch=configuration.get("check_val_every_n_epoch"),
            val_check_interval=configuration.get("val_check_interval"),
            gradient_clip_val=configuration.get("gradient_clip_val"),
            accumulate_grad_batches=configuration.get("accumulate_grad_batches"),
            precision=configuration.get("precision"),
            num_sanity_val_steps=0,
            logger=wandb_logger,
            log_every_n_steps=configuration.get("logging_steps", LOGGING_STEPS),
            # callbacks=[PushToHubCallback(MODEL_REPO_ID, args.subset), early_stop_callback]
            callbacks=[PushToHubCallback(MODEL_REPO_ID, args.subset)],
    )

    trainer.fit(idefics2)


def load_secret_dataset():

    val_datasets = []

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

    for subset in subsets_disponibles:
        val_dataset = Idefics2Dataset(processor=idefics2_processor, model=idefics2, dataset_name_or_path="de-Rodrigo/merit-secret", subset=subset, split="test", sort_json_key=False)

    val_dataset = ConcatDataset(val_datasets)

    return val_dataset


if __name__ == "__main__":

    # Define parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--subset", default=None, type=str)
    parser.add_argument("--save_initial", action="store_true", help="Save and upload vanilla model (PEFTed, no trained)")
    args = parser.parse_args()

    # Debug
    if eval(args.debug):
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to connect...")
        debugpy.wait_for_client()

    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    
    # Load Processor
    idefics2_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

    # Load Model
    idefics2 = load_model()

    # Load dataset partitions
    train_dataset = Idefics2Dataset(processor=idefics2_processor, model=idefics2, dataset_name_or_path=args.dataset, subset=args.subset, split="train", sort_json_key=False)
    val_dataset = Idefics2Dataset(processor=idefics2_processor, model=idefics2, dataset_name_or_path=args.dataset, subset=args.subset, split="validation", sort_json_key=False)
    # val_dataset = load_secret_dataset()
        
    # Apply Parameter-Efficient Fine-Tuning (PEFT)
    if not USE_ADD_ADAPTER:
        idefics2 = apply_peft()

    
    if args.save_initial:
        save_and_push_model(
            processor=idefics2_processor,
            model=idefics2,
            repo_id=MODEL_REPO_ID,
            dataset_subset="vanilla",
            save_dir="./initial_checkpoint",
            commit_message="Initial model upload (vanilla, encapsulated with PEFT)"
        )
        print("Initial model has been uploaded.")
        exit()

    # Collate Function
    image_token_id = idefics2_processor.tokenizer.additional_special_tokens_ids[idefics2_processor.tokenizer.additional_special_tokens.index("<image>")]

    # Callback
    # early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=4, verbose=False, mode="min")

    # Pytorch Lightning module
    idefics2_module, config = init_pl_module(idefics2_processor, idefics2)
    
    # Train
    train_idefics2(idefics2_module, config)
    