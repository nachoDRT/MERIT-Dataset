from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import HfApi
api = HfApi(endpoint="https://huggingface.co")

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
print(processor)