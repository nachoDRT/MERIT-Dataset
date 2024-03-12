from datasets import load_dataset, load_metric
from PIL import Image
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from torch.utils.data import DataLoader
from huggingface_hub import HfFolder
import numpy as np
import wandb
import json
import os

os.environ["WANDB_SILENT"] = "true"

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
HUGGINGFACE_LOGGING_PATH = "/app/config/huggingface_logging.json"

MAX_TRAIN_STEPS = 10000
EVAL_FRECUENCY = 250
LOGGING_STEPS = 1


class FunsdTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        train_dataset,
        validation_dataset,
        compute_metrics,
    ):
        super(FunsdTrainer, self).__init__(
            model=model,
            args=args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )

    def get_train_dataloader(self):
        return train_dataloader

    def get_test_dataloader(self, test_dataset):
        return test_dataloader


def preprocess_data(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    words = examples["words"]
    boxes = examples["bboxes"]
    word_labels = examples["ner_tags"]

    encoded_inputs = processor(
        images=images,
        text=words,
        boxes=boxes,
        word_labels=word_labels,
        padding="max_length",
        truncation=True,
    )
    return encoded_inputs


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


# Logging in wandb
with open(WANDB_LOGGING_PATH) as f:
    wandb_config = json.load(f)

    wandb.login()
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=wandb_config["name"],
        settings=wandb.Settings(console="off"),
    )

# # Logging in HuggingFace
# with open(HUGGINGFACE_LOGGING_PATH) as f:
#     hf_config = json.load(f)
#     HfFolder.save_token(hf_config["token"])

# Load dataset using a '.py' file
datasets = load_dataset(LOAD_DATASET_FROM_PY, trust_remote_code=True)

labels = datasets["train"].features["ner_tags"].feature.names

id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

# Load (pre) processor
processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", revision="no_ocr"
)

features = Features(
    {
        "image": Array3D(dtype="int64", shape=(3, 224, 224)),
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(Value(dtype="int64")),
        "token_type_ids": Sequence(Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

"""Preprocess datasets partitions"""
# Train
train_dataset = datasets["train"].map(
    preprocess_data,
    batched=True,
    remove_columns=datasets["train"].column_names,
    features=features,
)
train_dataset.set_format(type="torch")

# Validation
validation_dataset = datasets["validation"].map(
    preprocess_data,
    batched=True,
    remove_columns=datasets["validation"].column_names,
    features=features,
)
validation_dataset.set_format(type="torch")

# Test
test_dataset = datasets["test"].map(
    preprocess_data,
    batched=True,
    remove_columns=datasets["test"].column_names,
    features=features,
)
test_dataset.set_format(type="torch")

# Dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, pin_memory=False
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=2, shuffle=True, pin_memory=False
)
test_dataloader = DataLoader(test_dataset, batch_size=2, pin_memory=False)

model = LayoutLMv2ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", num_labels=len(label2id)
)

# Set id2label and label2id
model.config.id2label = id2label
model.config.label2id = label2id

# Metrics
metric = load_metric("seqeval")
return_entity_level_metrics = False

args = TrainingArguments(
    output_dir="".join(["app/", wandb_config["project"]]),
    max_steps=MAX_TRAIN_STEPS,
    # warmup_ratio=0.1,
    learning_rate=5e-5,
    fp16=True,
    push_to_hub=False,
    # push_to_hub_model_id="CICLAB-Comillas/layoutlmv2-LSD",
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_FRECUENCY,
    report_to="wandb",
    load_best_model_at_end=True,
    save_total_limit=1,
)

# Initialize our Trainer
trainer = FunsdTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Test
test_results = trainer.predict(test_dataset)

wandb.log(
    {
        "test_loss": test_results.metrics["test_loss"],
        "test_accuracy": test_results.metrics["test_accuracy"],
        "test_precision": test_results.metrics["test_precision"],
        "test_recall": test_results.metrics["test_recall"],
        "test_f1": test_results.metrics["test_f1"],
    }
)
wandb.finish()
