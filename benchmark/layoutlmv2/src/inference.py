import argparse
from datasets import load_dataset, ClassLabel, Sequence
import json
from torch.utils.data import DataLoader
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
)

GT_IS_PATH = False


def normalize_bbox(box, size):
    w, h = size
    return [
        int(1000 * box[0] / w),
        int(1000 * box[1] / h),
        int(1000 * box[2] / w),
        int(1000 * box[3] / h),
    ]

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


def load_session_dataset():

    split_test = {
        "test": "test[:1%]",
    }
        
    datasets_test = load_merit_dataset(testing_dataset_subset, split_test)

    return datasets_test

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--load_from_hub", action="store_true", default=False)
    parser.add_argument("--testing_dataset_subset", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()
    

    model_version = args.model_version
    load_from_hub = args.load_from_hub
    dataset_path = args.dataset_path
    testing_dataset_subset = args.testing_dataset_subset

    test_set = load_session_dataset()

    test_dataloader = DataLoader(test_set, batch_size=2, pin_memory=False)


    labels = test_set["test"].features["ner_tags"].feature.names
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        "de-Rodrigo/layoutlmv2", model_version, num_labels=len(label2id)
    )

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    encoded_inputs = processor(image, example['words'], boxes=example['bboxes'], word_labels=example['ner_tags'],
                            padding="max_length", truncation=True, return_tensors="pt")

    # Set id2label and label2id
    model.config.id2label = id2label
    model.config.label2id = label2id