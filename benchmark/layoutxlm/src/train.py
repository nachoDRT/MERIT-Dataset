import torch
import numpy as np
import json
import wandb
import os
import argparse
from typing import Optional, Union
from datasets import load_metric, load_dataset, ClassLabel, Sequence
from PIL import Image
import numpy as np
from huggingface_hub import login, HfApi
import io
from transformers import AutoTokenizer, TrainerCallback



os.environ["WANDB_SILENT"] = "true"

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

WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
DATASET_FOLDER = "/app/data/train-val/spanish/"

MAX_TRAIN_STEPS = 6000
EVAL_FRECUENCY = 250
LOGGING_STEPS = 1

HF_CARD_FILES = ["/app/src/card/README.md", "/app/src/card/.huggingface.yaml", "/app/src/card/assets/dragon_huggingface.png"]


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
            # del feature["id"]
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

def load_session_dataset():

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

def build_label_list():
    with open("assets/subjects_spanish.json") as f:
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


    return datasets


def load_image(path, size=224):
    # Igual que en tu script viejo
    img = Image.open(path).convert("RGB")
    w, h = img.size
    img_resized = img.resize((size, size))
    img_resized = np.asarray(img_resized)[:, :, ::-1].transpose(2, 0, 1)
    return img_resized, (w, h)


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


def normalize_bbox(b, size):

    b = simplify_bbox(b)
    
    return [
        int(1000 * b[0] / size[0]),
        int(1000 * b[1] / size[1]),
        int(1000 * b[2] / size[0]),
        int(1000 * b[3] / size[1]),
    ]


def load_and_resize(img, size=224):

    img = img.convert("RGB")
    original = img.copy()
    img = img.resize((size, size))
    # RGB → BGR y HWC → CHW
    arr = np.asarray(img)[:, :, ::-1].transpose(2, 0, 1)
    return arr, original

def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]

def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]

def add_layoutlm_fields(example):

    MAX_LEN = 512 


    img224, original = load_and_resize(example["image"])
    width, height = original.size


    doc  = json.loads(example["ground_truth"])
    lines = doc["form"]

    input_ids, bboxes, labels = [], [], []

    entities = []
    entity_id_to_index = {}
    id2label = {}

    for line in lines:
        if not line["words"]:
            continue

        word_texts = [w["text"] for w in line["words"]]
        word_boxes = [normalize_bbox(w["box"], (width, height))
                      for w in line["words"]]

        enc = tokenizer_data(
            word_texts,
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        word_ids = enc.word_ids()  

        start_pos = len(input_ids) 


        input_ids.extend(enc.input_ids)
        bboxes.extend([word_boxes[wid] for wid in word_ids])


        if line["label"] == "other":
            word_labels = ["O"] * len(word_texts)
        else:
            lbl = line["label"].upper()
            word_labels = [f"I-{lbl}"] * len(word_texts)
            word_labels[0] = f"B-{lbl}"

        token_labels = [LABEL2ID[word_labels[wid]] for wid in word_ids]
        labels.extend(token_labels)

        if word_labels[0] != "O":
            ent_idx = len(entities)
            entity_id_to_index[line["id"]] = ent_idx
            entities.append(
                {
                    "start": start_pos,
                    "end":   start_pos + len(enc.input_ids),
                    "label": line["label"].upper(),
                }
            )

        id2label[line["id"]] = line["label"]


    kv_rel = []
    for line in lines:
        for pair in line.get("linking", []):
            a, b = sorted(pair)
            if a in entity_id_to_index and b in entity_id_to_index:
                lbls = [id2label[a], id2label[b]]
                if lbls == ["question", "answer"]:
                    kv_rel.append(
                        {"head": entity_id_to_index[a], "tail": entity_id_to_index[b]}
                    )
                elif lbls == ["answer", "question"]:
                    kv_rel.append(
                        {"head": entity_id_to_index[b], "tail": entity_id_to_index[a]}
                    )

    def _span(rel):
        bounds = [
            entities[rel["head"]]["start"],
            entities[rel["head"]]["end"],
            entities[rel["tail"]]["start"],
            entities[rel["tail"]]["end"],
        ]
        return min(bounds), max(bounds)

    relations = [
        {
            "head": r["head"],
            "tail": r["tail"],
            "start_index": _span(r)[0],
            "end_index":   _span(r)[1],
        }
        for r in kv_rel
    ]
    relations.sort(key=lambda x: x["head"])


    if len(input_ids) > MAX_LEN:
        keep = MAX_LEN
        input_ids = input_ids[:keep]
        bboxes    = bboxes[:keep]
        labels    = labels[:keep]


        new_entities, old2new = [], {}
        for idx, ent in enumerate(entities):
            if ent["start"] < keep and ent["end"] <= keep:
                old2new[idx] = len(new_entities)
                new_entities.append(ent)
        entities = new_entities


        relations = [
            {
                "head":  old2new[rel["head"]],
                "tail":  old2new[rel["tail"]],
                "start_index": rel["start_index"],
                "end_index":   rel["end_index"],
            }
            for rel in relations
            if rel["head"] in old2new and rel["tail"] in old2new
        ]


    return {
        "input_ids":       input_ids,
        "bbox":            bboxes,
        "labels":          labels,
        "image":           img224, 
        "original_image":  original,
        "entities":        entities,
        "relations":       relations,
    }

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
    args = parser.parse_args()

    torch.cuda.empty_cache()

    feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
    tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
    tokenizer_data = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Logging in wandb
    with open(WANDB_LOGGING_PATH) as f:
        wandb_config = json.load(f)

    training_session_name = get_training_session_name(wandb_config)
    wandb.login()
    wandb.init(
        project=wandb_config["project"],
        entity=wandb_config["entity"],
        name=training_session_name,
    )

    if testing_dataset_subset:
        datasets_train_val, dataset_test = load_session_dataset()
    else:
        datasets_train_val, dataset_test = load_session_dataset()


    datasets_train_val = datasets_train_val.cast_column("labels", Sequence(ClassLabel(names=LABEL_LIST)))
    dataset_test = dataset_test.cast_column("labels", Sequence(ClassLabel(names=LABEL_LIST)))
    
    labels = datasets_train_val["train"].features["labels"].feature.names

    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}


    data_collator = DataCollatorForTokenClassification(
        feature_extractor,
        tokenizer,
        pad_to_multiple_of=None,
        padding="max_length",
        max_length=512,
    )

    train_dataset = datasets_train_val["train"]
    validation_dataset = datasets_train_val["validation"]
    test_dataset = dataset_test["test"]

    dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "microsoft/layoutxlm-base", id2label=id2label, label2id=label2id
    )

    # Metrics
    metric = load_metric("seqeval")
    return_entity_level_metrics = False


    args = TrainingArguments(
        output_dir="".join(["app/", wandb_config["project"]]),
        max_steps=MAX_TRAIN_STEPS,
        learning_rate=2.5e-5,
        # warmup_ratio=0.1,
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
        save_total_limit=1,
    )

    push_cb = PushToHubCallback(
        repo_id="de-Rodrigo/layoutxlm-merit",
        processor=tokenizer,
        monitor="eval_loss",
        mode="min",
        # save_dir="checkpoints"
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[push_cb],
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
