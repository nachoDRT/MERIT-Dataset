import wandb
import json
import os
from datasets import load_dataset, load_metric, Image as HFImage
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
import argparse
from huggingface_hub import login
import wandb
from PIL import Image


os.environ["WANDB_SILENT"] = "true"

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
HUGGINGFACE_LOGGING_PATH = "/app/config/huggingface_logging.json"
DATASET_FOLDER = "/app/data/train-val/english/"

MAX_TRAIN_STEPS = 10000
EVAL_FRECUENCY = 250
LOGGING_STEPS = 1


GT_IS_PATH = False


def normalize_bbox(box, size):
    w, h = size
    return [
        int(1000 * box[0] / w),
        int(1000 * box[1] / h),
        int(1000 * box[2] / w),
        int(1000 * box[3] / h),
    ]


def build_label_list():
    with open("assets/subjects_english.json") as f:
        sem = json.load(f)

    base = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
    for subj in sem["subjects"]:
        base += [f"B-{subj.upper()}", f"I-{subj.upper()}", f"B-{subj.upper()}_ANSWER", f"I-{subj.upper()}_ANSWER"]

    for tag in sem["academic_years_tags"]:
        yy = tag[1:].upper()
        base += [f"B-{yy}", f"I-{yy}"]

    for subj in sem["subjects"]:
        for tag in sem["academic_years_tags"]:
            key = f"{subj}{tag}".upper()
            base += [f"B-{key}", f"I-{key}", f"B-{key}_ANSWER", f"I-{key}_ANSWER"]
    return base


LABEL_LIST = build_label_list()
LABEL2ID = {t: i for i, t in enumerate(LABEL_LIST)}


def add_layoutlm_fields(example):

    # Parse ground-truth
    if GT_IS_PATH:
        with open(example["ground_truth"], "r", encoding="utf8") as f:
            data = json.load(f)
    else:
        data = json.loads(example["ground_truth"])

    # Image
    if isinstance(example["image"], dict):
        print(example["image"].keys())
        size = example["image"]["width"], example["image"]["height"]
        image_path = example["image"]["path"]
    else:

        img = example["image"]
        size = img.size
        image_path = getattr(img, "filename", None)

    # Annotations
    words, bboxes, ner_tags = [], [], []
    for item in data["form"]:
        words_example, label = item["words"], item["label"]
        words_example = [w for w in words_example if w["text"].strip() != ""]
        if len(words_example) == 0:
            continue
        if label == "other":
            for w in words_example:
                words.append(w["text"])
                ner_tags.append("O")
                bboxes.append(normalize_bbox(w["box"], size))
        else:
            words.append(words_example[0]["text"])
            ner_tags.append("B-" + label.upper())
            bboxes.append(normalize_bbox(words_example[0]["box"], size))
            for w in words_example[1:]:
                words.append(w["text"])
                ner_tags.append("I-" + label.upper())
                bboxes.append(normalize_bbox(w["box"], size))

        if load_from_hub:
            return {
                "words": words,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": img,
            }

        else:
            return {
                "words": words,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image_path": image_path,
            }


def get_dataset_name() -> str:
    dataset_name = ""

    for subset in os.listdir(DATASET_FOLDER):
        if dataset_name != "":
            dataset_name += "-"
        dataset_name += subset

    return dataset_name


def get_training_session_name(wandb_config: dict) -> str:

    if load_from_hub:
        dataset_name = dataset_subset
    else:
        dataset_name = get_dataset_name()
    name = "".join([wandb_config["name"], "_", dataset_name])

    return name


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

    if load_from_hub:
        images = examples["image"]
    else:
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


def init_apis():

    # Wandb
    with open(WANDB_LOGGING_PATH) as f:
        wandb_config = json.load(f)

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    training_session_name = get_training_session_name(wandb_config)

    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=training_session_name,
        settings=wandb.Settings(console="off"),
    )

    # HuggingFace
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    return wandb_config


def load_session_dataset():

    if load_from_hub:

        datasets = load_dataset(
            "de-Rodrigo/merit",
            name=dataset_subset,
            num_proc=16,
            split={
                "train": "train[:1%]",
                "validation": "validation[:1%]",
                "test": "test[:1%]",
            },
        )

        datasets = datasets.map(
            add_layoutlm_fields,
            batched=False,
            remove_columns=["ground_truth"],
        )

        class_label = ClassLabel(names=LABEL_LIST)
        datasets = datasets.cast_column("ner_tags", Sequence(class_label))

    else:
        # Load dataset using a '.py' file
        datasets = load_dataset(LOAD_DATASET_FROM_PY, trust_remote_code=True)

    return datasets


def get_dataset_partitions():
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

    return train_dataset, validation_dataset, test_dataset


def get_args():

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
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_hub", action="store_true", default=False)
    parser.add_argument("--dataset_subset", type=str, default=None)
    args = parser.parse_args()

    load_from_hub = args.load_from_hub
    dataset_subset = args.dataset_subset

    wandb_config = init_apis()
    datasets = load_session_dataset()

    labels = datasets["train"].features["ner_tags"].feature.names
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    # Load (pre) processor
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

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

    train_dataset, validation_dataset, test_dataset = get_dataset_partitions()

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=True, pin_memory=False)
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

    args = get_args()

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
