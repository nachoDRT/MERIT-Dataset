import re
import json
import random
from datasets import load_dataset
from typing import Any, List, Dict
from torch.utils.data import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

USE_LORA = False
USE_QLORA = True
USE_ADD_ADAPTER = True

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

        self.dataset = load_dataset(dataset_name_or_path, name=subset, split=self.split, num_proc=8)
        self.dataset_length = len(self.dataset)

        self.gt_token_sequences = []
        self.added_tokens = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(
                    ground_truth["gt_parse"], dict
                )
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

    def json2token(
        self,
        obj: Any,
        update_special_tokens_for_json_key: bool = True,
        sort_json_key: bool = True,
    ):
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
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(
                            obj[k], update_special_tokens_for_json_key, sort_json_key
                        )
                        + rf"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [
                    self.json2token(
                        item, update_special_tokens_for_json_key, sort_json_key
                    )
                    for item in obj
                ]
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
    
    def get_model(self):
        print(f"ADDED TOKENS: {self.add_tokens}")
        print(f"ADDED TOKENS: {len(self.added_tokens)}")

        if not USE_ADD_ADAPTER:
            lora_config = LoraConfig(
                    r=8,
                    lora_alpha=8,
                    lora_dropout=0.1,
                    target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
                    use_dora=False if USE_QLORA else True,
                    init_lora_weights="gaussian",
                )

            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)

        return self.model
    
    def get_processor(self):
        return self.processor


def token2json(tokens, processor, is_inner_value=False, added_vocab=None):
    """
    Convert a (generated) token sequence into an ordered JSON format.
    """
    if added_vocab is None:
        added_vocab = processor.tokenizer.get_added_vocab()

    output = {}

    while tokens:
        start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.group(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, "")
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f"{start_token_escaped}(.*?){end_token_escaped}",
                tokens,
                re.IGNORECASE | re.DOTALL,
            )
            if content is not None:
                content = content.group(1).strip()
                if r"<s_" in content and r"</s_" in content:  # non-leaf node
                    value = token2json(
                        content, processor, is_inner_value=True, added_vocab=added_vocab
                    )
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  # leaf nodes
                    output[key] = []
                    for leaf in content.split(r"<sep/>"):
                        leaf = leaf.strip()
                        if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                            leaf = leaf[1:-2]  # for categorical special tokens
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]

            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r"<sep/>":  # non-leaf nodes
                return [output] + token2json(
                    tokens[6:], processor, is_inner_value=True, added_vocab=added_vocab
                )

    if len(output):
        return [output] if is_inner_value else output
    else:
        return [] if is_inner_value else {"text_sequence": tokens}
