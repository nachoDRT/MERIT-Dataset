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
from torch.utils.data import DataLoader, ConcatDataset
from huggingface_hub import HfFolder
import numpy as np
import argparse
from huggingface_hub import login
import wandb
from PIL import Image
from transformers import TrainerCallback
import os
from huggingface_hub import login, HfApi


os.environ["WANDB_SILENT"] = "true"

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
HUGGINGFACE_LOGGING_PATH = "/app/config/huggingface_logging.json"
DATASET_FOLDER = "/app/data/train-val/english/"

MAX_TRAIN_STEPS = 6000
EVAL_FRECUENCY = 250
LOGGING_STEPS = 1


GT_IS_PATH = False
HF_CARD_FILES = ["/app/src/card/README.md", "/app/src/card/.huggingface.yaml", "/app/src/card/assets/dragon_huggingface.png"]



class PushToHubCallback(TrainerCallback):
    def __init__(
        self,
        repo_id: str,
        processor,
        monitor: str = "eval_loss",
        mode: str = "min",
        commit_message: str = "Improved model",
        use_auth_token: str = True,
    ):
        self.repo_id = repo_id
        self.processor = processor
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.commit_message = commit_message
        self.use_auth_token = use_auth_token
        login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
        self.api = HfApi()

        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' o 'max'")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current = metrics.get(self.monitor)
        if current is None:
            return

        improved = (current < self.best_score) if self.mode == "min" else (current > self.best_score)
        if not improved:
            return

        print(f"[PushToHubCallback] {self.monitor} mejoró: {self.best_score:.4f} → {current:.4f}")
        self.best_score = current

        kwargs["model"].push_to_hub(
            repo_id=self.repo_id,
            commit_message=f"{self.commit_message}: step {state.global_step} ({self.monitor}={current:.4f})",
            use_auth_token=self.use_auth_token
        )

        self.processor.push_to_hub(
            repo_id=self.repo_id,
            commit_message=f"{self.commit_message} (processor): step {state.global_step}",
            use_auth_token=self.use_auth_token
        )

        for file_path in HF_CARD_FILES:
            if os.path.isfile(file_path):
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=self.repo_id,
                    repo_type="model",
                    commit_message=f"Añadiendo {os.path.basename(file_path)}"
                )
            else:
                print(f"[PushToHubCallback] Warning: {file_path} does not exist.")

        kwargs["model"].train()


def simplify_bbox(bbox):

    if len(bbox) == 4:
        xs = bbox[0::2]  # [x0, x1]
        ys = bbox[1::2]  # [y0, y1]
    elif len(bbox) == 8:
        xs = bbox[0::2]  # [x0, x1, x2, x3]
        ys = bbox[1::2]  # [y0, y1, y2, y3]
    else:
        raise ValueError("List must be 4 or 8 values")

    return [min(xs), min(ys), max(xs), max(ys)]

def normalize_bbox(box, size):

    box = simplify_bbox(box)
    
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
    # ---------- 1. Parse ground-truth ----------
    if GT_IS_PATH:
        with open(example["ground_truth"], "r", encoding="utf8") as f:
            data = json.load(f)
    else:
        data = json.loads(example["ground_truth"])

    # ---------- 2. Imagen ----------
    if isinstance(example["image"], dict):        # caso “imagen serializada” (p.ej. en HF Hub)
        size = (example["image"]["width"], example["image"]["height"])
        image_path = example["image"]["path"]
        img = None                                # no tenemos el objeto PIL
    else:                                         # caso PIL.Image
        img = example["image"].convert("RGB")
        size = img.size
        image_path = getattr(img, "filename", None)

    # ---------- 3. Anotaciones ----------
    words, bboxes, ner_tags = [], [], []

    for item in data["form"]:
        words_example, label = item["words"], item["label"]
        words_example = [w for w in words_example if w["text"].strip() != ""]
        if not words_example:
            continue

        if label == "other":
            for w in words_example:
                words.append(w["text"])
                ner_tags.append("O")
                bboxes.append(normalize_bbox(w["box"], size))
        else:
            # Etiqueta B- para la primera palabra
            words.append(words_example[0]["text"])
            ner_tags.append("B-" + label.upper())
            bboxes.append(normalize_bbox(words_example[0]["box"], size))
            # Etiquetas I- para las siguientes
            for w in words_example[1:]:
                words.append(w["text"])
                ner_tags.append("I-" + label.upper())
                bboxes.append(normalize_bbox(w["box"], size))

    # ---------- 4. Ensamblar y devolver ----------
    features = {
        "words": words,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
    }

    if load_from_hub:
        # Cuando se entrena directamente desde HF Hub suele necesitar el objeto PIL
        features["image"] = img if img is not None else example["image"]
    else:
        features["image_path"] = image_path

    return features



def get_dataset_name() -> str:
    dataset_name = ""

    for subset in os.listdir(DATASET_FOLDER):
        if dataset_name != "":
            dataset_name += "-"
        dataset_name += subset

    return dataset_name


def get_training_session_name(wandb_config: dict) -> str:

    if load_from_hub:
        dataset_name = training_dataset_subset
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
        callbacks=None,
    ):
        super(FunsdTrainer, self).__init__(
            model=model,
            args=args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            # callbacks=callbacks,
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
        max_length=512,
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

        split = {
            "train": "train[:100%]",
            "validation": "validation[:5%]",
            "test": "test[:5%]",
        }

        if dataset_path == "de-Rodrigo/merit" and testing_dataset_subset == None:
            datasets = load_merit_dataset(training_dataset_subset, split)
        
        elif dataset_path == "de-Rodrigo/merit" and testing_dataset_subset != None:
            
            split_train_val = {
                "train": "train[:100%]",
                "validation": "validation[:5%]",
            }

            split_test = {
                "test": "test[:5%]",
            }

            datasets_train_val = load_merit_dataset(training_dataset_subset, split_train_val)
            datasets_test = load_merit_dataset(testing_dataset_subset, split_test)

            return datasets_train_val, datasets_test

    else:
        # Load dataset using a '.py' file
        datasets = load_dataset(LOAD_DATASET_FROM_PY, trust_remote_code=True)

    return datasets, None


def load_merit_dataset(dataset_subset, split):
    datasets = load_dataset(
        path=dataset_path,
        name=dataset_subset,
        num_proc=16,
        split=split
    )

    datasets = datasets.map(
        add_layoutlm_fields,
        batched=False,
        remove_columns=["ground_truth"],
    )

    class_label = ClassLabel(names=LABEL_LIST)
    datasets = datasets.cast_column("ner_tags", Sequence(class_label))

    return datasets


def get_dataset_partitions(datasets_train_val, datasets_test):
    """Preprocess datasets partitions"""
    # Train
    train_dataset = datasets_train_val["train"].map(
        preprocess_data,
        batched=True,
        remove_columns=datasets_train_val["train"].column_names,
        features=features,
    )
    train_dataset.set_format(type="torch")

    # Validation
    validation_dataset = datasets_train_val["validation"].map(
        preprocess_data,
        batched=True,
        remove_columns=datasets_train_val["validation"].column_names,
        features=features,
    )
    validation_dataset.set_format(type="torch")

    if datasets_test:
        test_dataset = datasets_test["test"].map(
            preprocess_data,
            batched=True,
            remove_columns=datasets_test["test"].column_names,
            features=features,
        )
        test_dataset.set_format(type="torch")
    
    else:
        # Test
        test_dataset = datasets_train_val["test"].map(
            preprocess_data,
            batched=True,
            remove_columns=datasets_train_val["test"].column_names,
            features=features,
        )
        test_dataset.set_format(type="torch")

    return train_dataset, validation_dataset, test_dataset


def get_args():

    args = TrainingArguments(
        output_dir="".join(["app/", wandb_config["project"]]),
        max_steps=MAX_TRAIN_STEPS,
        # warmup_ratio=0.1,
        learning_rate=1.5e-5,
        fp16=True,
        push_to_hub=False,
        # push_to_hub_model_id="CICLAB-Comillas/layoutlmv2-LSD",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_FRECUENCY,
        save_steps=EVAL_FRECUENCY,
        report_to="wandb",
        load_best_model_at_end=True,
        save_total_limit=1,
    )
    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_from_hub", action="store_true", default=False)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--training_dataset_subset", type=str, default=None)
    parser.add_argument("--testing_dataset_subset", type=str, default=None)
    args = parser.parse_args()

    load_from_hub = args.load_from_hub
    dataset_path = args.dataset_path
    training_dataset_subset = args.training_dataset_subset
    testing_dataset_subset = args.testing_dataset_subset

    wandb_config = init_apis()

    if testing_dataset_subset:
        datasets_train_val, dataset_test = load_session_dataset()
    else:
        datasets_train_val, dataset_test = load_session_dataset()

    # ex = datasets_train_val["train"][0]
    # print(ex)

    # ex = dataset_test["test"][0]
    # print(ex)
    
    labels = datasets_train_val["train"].features["ner_tags"].feature.names
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

    train_dataset, validation_dataset, test_dataset = get_dataset_partitions(datasets_train_val, dataset_test)

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

    push_cb = PushToHubCallback(
        repo_id="de-Rodrigo/layoutlmv2-merit",
        processor=processor,
        monitor="eval_loss",
        mode="min",
        # save_dir="checkpoints"
    )

    # Initialize our Trainer
    trainer = FunsdTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[push_cb],
    )

    # Train
    trainer.train()

    # Test
    test_results = trainer.predict(test_dataset)

    preds, labels = test_results.predictions, test_results.label_ids
    metric = load_metric("seqeval")

    true_f1s = []
    for pred_logits, label_ids in zip(preds, labels):
        pred_ids = np.argmax(pred_logits, axis=-1)
        true_labels, true_preds = [], []
        for p, l in zip(pred_ids, label_ids):
            if l != -100:
                true_labels.append(id2label[l])
                true_preds.append(id2label[p])
        result = metric.compute(predictions=[true_preds], references=[true_labels])
        true_f1s.append(result["overall_f1"])

    # Convertir a lista y mostrarla:
    f1_list = list(true_f1s)
    print(f1_list)

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
