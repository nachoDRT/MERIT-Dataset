---
license: mit
datasets:
- de-Rodrigo/merit
language:
- en
- es
base_model:
- naver-clova-ix/donut-base-finetuned-cord-v2
pipeline_tag: image-text-to-text
---

# DONUT Merit

<a href="https://x.com/nearcyan/status/1706914605262684394">
  <div style="text-align: center;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/de-Rodrigo/donut-merit/resolve/main/assets/dragon_huggingface.png">
      <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/de-Rodrigo/donut-merit/resolve/main/assets/dragon_huggingface.png">
      <img alt="DragonHuggingFace" src="https://huggingface.co/de-Rodrigo/donut-merit/resolve/main/assets/dragon_huggingface.png" style="width: 200px;">
    </picture>
  </div>
</a>


## Model Architecture
**This model is based on the Donut architecture and fine-tuned on the Merit dataset for form understanding tasks.**

- Backbone: [Donut](https://huggingface.co/naver-clova-ix/donut-base)
- Training Data: [Merit](https://huggingface.co/datasets/de-Rodrigo/merit)

## Example Usage

```python
from transformers import AutoTokenizer, AutoModelForImageTextToText

tokenizer = AutoTokenizer.from_pretrained("de-Rodrigo/donut-merit")
model = AutoModelForImageTextToText.from_pretrained("de-Rodrigo/donut-merit")
```
**WIP** 🛠️