from datasets import load_dataset
from transformers import AutoProcessor
from datasets.features import ClassLabel
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import load_metric
import numpy as np
from transformers import LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import json
import wandb
import os
import argparse

os.environ["WANDB_SILENT"] = "true"

LOAD_DATASET_FROM_PY = "/app/src/load_dataset.py"
WANDB_LOGGING_PATH = "/app/config/wandb_logging.json"
DATASET_FOLDER = "/app/data/train-val/english/"

MAX_TRAIN_STEPS = 10000
EVAL_FRECUENCY = 250
LOGGING_STEPS = 1


def get_training_session_name(wandb_config: dict) -> str:

    dataset_name = training_dataset_subset
    name = "".join([wandb_config["name"], "_", dataset_name])

    return name


def prepare_examples(examples):

    images = examples[image_column_name]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
    )

    return encoding


def get_label_list(labels):

    unique_labels = set()

    for label in labels:
        unique_labels = unique_labels | set(label)

    label_list = list(unique_labels)
    label_list.sort()

    return label_list


def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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

    if load_from_hub:
        
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

            

def load_same_session_dataset():
    
    if load_from_hub:
    
        split_train_val = {
            "train": "train[:100%]",
            "validation": "validation[:5%]",
            "test": "test[:5%]",
        }

        datasets_train_val_test = load_merit_dataset(training_dataset_subset, split_train_val)

        return datasets_train_val_test


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

def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]

    return bbox


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def add_layoutlm_fields(example, guid):
    
    tokens = []
    bboxes = []
    ner_tags = []

    # file_path = os.path.join(ann_dir, file)
    # with open(file_path, "r", encoding="utf8") as f:
    #     data = json.load(f)
    
    image = example["image"]
    image = image.convert("RGB")
    size = image.size
    
    
    # image_path = os.path.join(img_dir, file)
    # image_path = image_path.replace("json", "png")
    # image, size = load_image(image_path)
    data = json.loads(example["ground_truth"])
    for item in data["form"]:
        cur_line_bboxes = []
        words, label = item["words"], item["label"]
        words = [w for w in words if w["text"].strip() != ""]
        if len(words) == 0:
            continue
        if label == "other":
            for w in words:
                tokens.append(w["text"])
                ner_tags.append("O")
                cur_line_bboxes.append(normalize_bbox(w["box"], size))
        else:
            tokens.append(words[0]["text"])
            ner_tags.append("B-" + label.upper())
            cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
            for w in words[1:]:
                tokens.append(w["text"])
                ner_tags.append("I-" + label.upper())
                cur_line_bboxes.append(normalize_bbox(w["box"], size))
        cur_line_bboxes = get_line_bbox(cur_line_bboxes)
        bboxes.extend(cur_line_bboxes)

    return {
        "id": str(guid),
        "tokens": tokens,
        "bboxes": bboxes,
        "ner_tags": ner_tags,
        "image": image,
    }

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
        with_indices=True,
        remove_columns=["ground_truth"],
    )

    class_label = ClassLabel(names=LABEL_LIST)
    datasets = datasets.cast_column("ner_tags", Sequence(class_label))

    return datasets


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

    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    if testing_dataset_subset != None:
        datasets_train_val, dataset_test = load_session_dataset()
        dataset_train = datasets_train_val["train"]
        dataset_validation = datasets_train_val["validation"]
        dataset_test = dataset_test["test"]
    
    elif testing_dataset_subset == None:
        datasets_train_val_test = load_same_session_dataset()
        dataset_train = datasets_train_val_test["train"]
        dataset_validation = datasets_train_val_test["validation"]
        dataset_test = datasets_train_val_test["test"]

    # dataset = load_dataset(LOAD_DATASET_FROM_PY)

    features = dataset_train.features
    column_names = dataset_train.column_names
    image_column_name = "image"
    text_column_name = "tokens"
    boxes_column_name = "bboxes"
    label_column_name = "ner_tags"


    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
    else:
        label_list = get_label_list(dataset_train[label_column_name])
        id2label = {k: v for k, v in enumerate(label_list)}
        label2id = {v: k for k, v in enumerate(label_list)}
    num_labels = len(label_list)


    features = Features(
        {
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(feature=Value(dtype="int64")),
        }
    )

    train_dataset = dataset_train.map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )
    eval_dataset = dataset_validation.map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    test_dataset = dataset_test.map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features,
    )

    metric = load_metric("seqeval")
    return_entity_level_metrics = False

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="".join(["app/", wandb_config["project"]]),
        max_steps=MAX_TRAIN_STEPS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2.5e-5,
        push_to_hub=False,
        # push_to_hub_model_id="CICLAB-Comillas/layoutlmv3-LSD",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=EVAL_FRECUENCY,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
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
