import argparse
from huggingface_hub import login, HfApi
from datasets import load_dataset
import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Any, List, Dict
import random
import json
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import lightning as L
import torch
import re
import wandb
from nltk import edit_distance
import numpy as np
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import precision_score, recall_score, f1_score
from torchmetrics.classification import MulticlassF1Score
from PIL import Image
import io
import numpy as np
import copy

HF_CARD_FILES = [
    "/app/src/card/README.md",
    "/app/src/card/.huggingface.yaml",
    "/app/src/card/assets/dragon_huggingface.png",
]

REPO_ID = "google/paligemma-3b-pt-224"
MAX_LENGTH = 512
WANDB_PROJECT = "MERIT-Dataset-Img2Sequence"


PROMPT = "extract JSON."


class CustomDataset(Dataset):
    """
    PyTorch Dataset. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        subset: str,
        split: str = "train",
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.split = split
        self.sort_json_key = sort_json_key

        if dataset_name_or_path == "de-Rodrigo/merit":
            self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split, num_proc=8)
        else:
            self.dataset = load_dataset(dataset_name_or_path, split=self.split, num_proc=8)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []

        if dataset_name_or_path == "de-Rodrigo/merit":
            for sample in self.dataset:
                ground_truth = json.loads(sample["ground_truth"])
                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons  # load json from list of json
                    ]
                )
        else:
            for sample in self.dataset:
                ocr_words = sample["ocr_words"]
                words_list = [{"word": word} for word in ocr_words]
                page = {"page_0": words_list}
                ground_truth = {"gt_parse": page}

                if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                    assert isinstance(ground_truth["gt_parses"], list)
                    gt_jsons = ground_truth["gt_parses"]
                else:
                    assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                    gt_jsons = [ground_truth["gt_parse"]]

                self.gt_token_sequences.append(
                    [
                        self.json2token(
                            gt_json,
                            sort_json_key=self.sort_json_key,
                        )
                        for gt_json in gt_jsons  # load json from list of json
                    ]
                )


    def json2token(self, obj: Any, sort_json_key: bool = True):
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
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

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

        # inputs
        image = sample["image"]
        target_sequence = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1

        return image, target_sequence


class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

        self._all_preds = []
        self._all_labels = []

        vocab_size = processor.tokenizer.vocab_size
        pad_id     = processor.tokenizer.pad_token_id
        self.test_f1 = MulticlassF1Score(
            num_classes=vocab_size,
            average="macro",
            ignore_index=pad_id,
        )
        self.pad_id = pad_id


    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values, answers = batch

        dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(dtype)

        # autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
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
    

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, answers = batch
        device = self.device
        dtype  = next(self.model.parameters()).dtype

        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values   = pixel_values.to(device=device, dtype=dtype)

        # tokenize the references
        encoding = self.processor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        labels = encoding.input_ids           # [B, L₂]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        logits = outputs.logits               # [B, L₁, V]
        preds  = torch.argmax(logits, dim=-1) # [B, L₁]

        # 1) build mask where labels ≠ pad
        mask   = labels != self.pad_id       # [B, L₂]

        # 2) trim mask to preds’ length if needed
        mask   = mask[:, : preds.size(1)]    # now [B, L₁]

        # 3) flatten out only the real tokens
        flat_preds  = preds[mask].flatten()  # 1D
        flat_labels = labels[:, : preds.size(1)][mask].flatten()  # 1D

        # 4) update the metric
        self.test_f1(flat_preds, flat_labels)

        return None

    def on_test_epoch_end(self):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1, prog_bar=True)

    def on_test_start(self) -> None:
        # al arrancar test: reinicia las listas
        self._all_preds.clear()
        self._all_labels.clear()

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    # def val_dataloader(self):
    #     return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        # Muestreamos solo un 10% de val_dataset
        percentage = 0.1
        num_samples = int(len(val_dataset) * percentage)
        # índices aleatorios sin reemplazo
        indices = torch.randperm(len(val_dataset))[:num_samples].tolist()
        sampler = SubsetRandomSampler(indices)

        return DataLoader(
            val_dataset,
            sampler=sampler,
            collate_fn=eval_collate_fn,
            batch_size=self.batch_size,
            num_workers=4,
        )


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
        self.best_score = float('inf') if mode == "min" else -float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        improved = (
            current_score < self.best_score if self.mode == "min"
            else current_score > self.best_score
        )
        if improved:
            print(f"Detected improvement in {self.monitor}: {self.best_score} -> {current_score}")
            self.best_score = current_score
            self._push_model(trainer, pl_module, epoch=trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        print("Uploading final card files to the Hub...")
        repo_id = f"de-Rodrigo/{self.model_output_name}"
        self._upload_card_files(repo_id)

    def _push_model(self, trainer, pl_module, epoch: int):
        # 1) Construye la ruta donde guardar
        save_path = os.path.join(
            self.save_dir,
            f"{self.model_output_name}_{self.dataset_subset}_epoch{epoch}",
        )

        # 2) Duplica el modelo en CPU para no saturar la GPU con la copia
        temp_model = copy.deepcopy(pl_module.model).to("cpu")

        # 3) Fusiona LoRA sobre la copia sin tocar pl_module.model
        if isinstance(temp_model, PeftModel):
            model_to_save = temp_model.merge_and_unload()
        else:
            model_to_save = temp_model

        # 4) Guarda el modelo fusionado completo + processor/tokenizer
        model_to_save.save_pretrained(
            save_path,
            safe_serialization=True,
            max_shard_size="1GB",
        )
        pl_module.processor.save_pretrained(save_path)

        # 5) Guarda el adapter LoRA para que luego PeftModel lo cargue desde este subfolder
        #    Esto creará adapter_config.json + pytorch_model.bin (solo los pesos LoRA)
        if isinstance(pl_module.model, PeftModel):
            pl_module.model.save_pretrained(save_path)

        # 6) Sube todo al Hub bajo el subfolder correspondiente
        repo_id = f"de-Rodrigo/{self.model_output_name}"
        self.api.upload_folder(
            folder_path=save_path,
            path_in_repo=self.dataset_subset,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Best model up to epoch {epoch} ({self.monitor}={self.best_score})",
        )
        self._upload_card_files(repo_id)

        # 7) Libera la copia y vacía caché de CUDA
        del temp_model, model_to_save
        torch.cuda.empty_cache

    def _upload_card_files(self, repo_id: str):
        for file_path in HF_CARD_FILES:
            print(f"Uploading {file_path} to {repo_id}")
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo="/".join(file_path.split(os.sep)[4:]),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Uploading card files",
            )


def train_collate_fn(examples):
    images = [example[0] for example in examples]
    texts = [PROMPT for _ in range(len(images))]
    labels = [example[1] for example in examples]

    # Normalize all images to PIL RGB
    imgs_processed = []
    for img in images:
        # If bytes (e.g. raw JPEG data)
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        # If file path
        elif isinstance(img, str):
            img = Image.open(img)
        # If numpy array
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img).convert("RGB")
            elif img.ndim == 3:
                img = Image.fromarray(img)
        # Ensure in RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgs_processed.append(img)

    # Build inputs
    inputs = processor(
        text=texts,
        images=imgs_processed,
        suffix=labels,
        return_tensors="pt",
        padding=True,
        truncation="only_second",
        max_length=MAX_LENGTH,
        tokenize_newline_separately=False
    )

    return (
        inputs["input_ids"],
        inputs["token_type_ids"],
        inputs["attention_mask"],
        inputs["pixel_values"],
        inputs["labels"],
    )


def eval_collate_fn(examples):
    images = [example[0] for example in examples]
    texts = [PROMPT for _ in range(len(images))]
    answers = [example[1] for example in examples]

    # Normalize all images to PIL RGB
    imgs_processed = []
    for img in images:
        if isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        elif isinstance(img, str):
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = Image.fromarray(img).convert("RGB")
            elif img.ndim == 3:
                img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        imgs_processed.append(img)

    # Build inputs
    inputs = processor(
        text=texts,
        images=imgs_processed,
        return_tensors="pt",
        padding=True,
        tokenize_newline_separately=False
    )

    return (
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["pixel_values"],
        answers,
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_subsets", type=str)
    # parser.add_argument("--test_dataset_version", type=str)
    # parser.add_argument("--dataset_subsets", type=str, action='append')
    # parser.add_argument("--freeze_encoder", action="store_true", default=False)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_subsets = args.dataset_subsets
    # test_dataset_version = args.test_dataset_version
    model_output_name = "".join(["paligemma-", dataset_name.split('/')[-1]])


    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    # Q-LoRa
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID, quantization_config=bnb_config, device_map={"":0})
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
   
    session_name = "_".join([dataset_subsets])
    session_name = f"paligemma_{session_name}"

    train_dataset = CustomDataset(dataset_name, dataset_subsets, split="train")
    val_dataset = CustomDataset(dataset_name, dataset_subsets, split="validation")


    # test_dataset = CustomDataset(dataset_name, test_dataset_version, split="test")

    processor = AutoProcessor.from_pretrained(REPO_ID)

    config = {
        "max_epochs": 5,
        "val_check_interval": 0.2,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }

    model_module = PaliGemmaModelPLModule(config, processor, model)

    # early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=session_name, entity="iderodrigo")

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        # precision="16-mixed",
        precision="bf16-mixed",
        limit_val_batches=20,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[PushToHubCallback("paligemma-merit", session_name)],
        log_every_n_steps=1,
        val_check_interval=config["val_check_interval"],
    )

    trainer.fit(model_module)
    wandb.finish()
