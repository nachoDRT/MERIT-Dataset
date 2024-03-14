"""
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
"""

import json
import os
from PIL import Image
import datasets
from pathlib import Path
from tqdm import tqdm

FUNSD_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
# FUNSD = True

SUBJECTS_SEMANTIC_JSON = "/app/assets/subjects_english.json"
ZIP_FILE_PATH = "/app/data/dataset.zip"


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for LSD in FUNSD format"""

    def __init__(self, **kwargs):
        """BuilderConfig for LSD in FUNSD format.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """LSD dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(
            name="LSD",
            version=datasets.Version("1.0.0"),
            description="LSD dataset in FUNSD format for LayoutLMv3",
        ),
    ]

    def get_dataset_labels(self):

        # Open '.json' file with info about subjects' synonyms
        with open(SUBJECTS_SEMANTIC_JSON, "r") as subjects_json_file:
            data = subjects_json_file.read()
            subjects_semantic = json.loads(data)

        subjects_json_file.close()

        lsd_labels = FUNSD_LABELS
        # Loop over the dictionary to extract its keys
        # if not FUNSD:

        for key in subjects_semantic["subjects"]:

            # Construct labels strings and append
            b_label = "".join(["B-", key.upper()])
            lsd_labels.append(b_label)
            i_label = "".join(["I-", key.upper()])
            lsd_labels.append(i_label)
            b_answer_label = "".join([b_label, "_ANSWER"])
            lsd_labels.append(b_answer_label)
            i_answer_label = "".join([i_label, "_ANSWER"])
            lsd_labels.append(i_answer_label)

        return lsd_labels

    def _expand_labels_with_academic_years(
        self, lsd_labels: list, academic_years: list, names_labels: list
    ):

        for academic_year in academic_years:

            names_labels.append(academic_year[1:].upper())
            lsd_labels.append("".join(["B-", academic_year[1:].upper()]))
            lsd_labels.append("".join(["I-", academic_year[1:].upper()]))

        return lsd_labels, names_labels

    def get_dataset_labels_v2(self):

        # Open '.json' file with info about subjects' synonyms
        with open(SUBJECTS_SEMANTIC_JSON, "r") as subjects_json_file:
            data = subjects_json_file.read()
            subjects_semantic = json.loads(data)

        subjects_json_file.close()

        lsd_labels = FUNSD_LABELS
        names_labels = []
        # Loop over the dictionary to extract its keys
        expanded_labels = []
        expanded_tags = subjects_semantic["academic_years_tags"]
        lsd_labels, names_labels = self._expand_labels_with_academic_years(
            lsd_labels, expanded_tags, names_labels
        )

        for key in subjects_semantic["subjects"]:
            for tag in expanded_tags:
                expanded_labels.append("".join([key, tag]))

        for key in expanded_labels:

            # Construct labels strings and append
            names_labels.append(key.upper())
            names_labels.append("".join([key.upper(), "_ANSWER"]))
            b_label = "".join(["B-", key.upper()])
            lsd_labels.append(b_label)
            i_label = "".join(["I-", key.upper()])
            lsd_labels.append(i_label)
            b_answer_label = "".join([b_label, "_ANSWER"])
            lsd_labels.append(b_answer_label)
            i_answer_label = "".join([i_label, "_ANSWER"])
            lsd_labels.append(i_answer_label)

        return lsd_labels, names_labels

    def _info(self):

        lsd_labels, _ = self.get_dataset_labels_v2()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"))
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(names=lsd_labels)
                    ),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # zip_file_path = os.path.join(os.getcwd(), "dataset.zip")
        downloaded_file = dl_manager.extract(ZIP_FILE_PATH)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": f"{downloaded_file}/training_data/"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": f"{downloaded_file}/validating_data/"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": f"{downloaded_file}/testing_data/"},
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]

        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
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
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)

            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": image,
            }
