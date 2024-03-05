import torch
from typing import Optional, Union
from datasets import load_metric, load_dataset
import numpy as np
import json
import wandb

from transformers import (
    LayoutLMv2ForTokenClassification,
    LayoutLMv2FeatureExtractor,
    LayoutXLMTokenizer,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)

from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader
from dataclasses import dataclass

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
HUGGINGFACE_LOGGING_PATH = "/app/config/huggingface_logging.json"

MAX_TRAIN_STEPS = 150
EVAL_FRECUENCY = 100
LOGGING_STEPS = 50

torch.cuda.empty_cache()

# Logging in wandb
with open(WANDB_LOGGING_PATH) as f:
    wandb_config = json.load(f)

    wandb.login()
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=wandb_config["name"],
    )

# Load dataset using a '.py' file
dataset = load_dataset(LOAD_DATASET_FROM_PY, "xfun_es")
dataset.shuffle()

labels = dataset["train"].features["labels"].feature.names

id2label = {k: v for k, v in enumerate(labels)}
label2id = {v: k for k, v in enumerate(labels)}

feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor(
            [feature["original_image"] for feature in features], return_tensors="pt"
        ).pixel_values

        # prepare text input
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["original_image"]
            del feature["entities"]
            del feature["relations"]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["image"] = image

        return batch


data_collator = DataCollatorForTokenClassification(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=None,
    padding="max_length",
    max_length=512,
)

train_dataset = dataset["train"]
test_dataset = dataset["validation"]

dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

model = LayoutLMv2ForTokenClassification.from_pretrained(
    "microsoft/layoutxlm-base", id2label=id2label, label2id=label2id
)

# Metrics
metric = load_metric("seqeval")
return_entity_level_metrics = True


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# args = TrainingArguments(
#     output_dir="layoutxlm-finetuned-es_trunds",  # name of directory to store the checkpoints
#     overwrite_output_dir=True,
#     max_steps=TOTAL_STEPS,  # we train for a maximum of 1,000 batches
#     warmup_ratio=0.1,  # we warmup a bit
#     # fp16=True, # we use mixed precision (less memory consumption)
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     learning_rate=1e-5,
#     remove_unused_columns=False,
#     push_to_hub=False,  # we'd like to push our model to the hub during training
#     save_steps=SAVE_STEPS,
#     save_total_limit=1,
# )

args = TrainingArguments(
    output_dir=wandb_config["project"],
    max_steps=MAX_TRAIN_STEPS,
    warmup_ratio=0.1,
    fp16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    push_to_hub=False,
    # push_to_hub_model_id="CICLAB-Comillas/layoutlmv2-LSD",
    remove_unused_columns=False,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_FRECUENCY,
    report_to="wandb",
    load_best_model_at_end=True,
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# trainer.train(frequency_to_evaluate=F_TO_EVALUATE)
trainer.train()
wandb.finish()
