---
base_model: facebook/nllb-200-distilled-600M
library_name: peft
pipeline_tag: translation
tags:
- translation
- twi
- english
- nllb
- peft
- lora
language:
- tw
- en
license: apache-2.0
---

# Twi to English NLLB LoRA Adapter

This repository contains a LoRA adapter fine-tuned for Twi to English translation.
It was trained from `facebook/nllb-200-distilled-600M` using the local Ghana NLP
Twi-English parallel text dataset included in the Sukuupath project.

## Project Collaboration

This model was built and first published by **McLanor Jeff** under the personal
repository `mclanorjeff/twi-english-nllb-lora`.

The shared collaboration version for the project team is available under the
**Lanor-and-Nick** organization:

- `Lanor-and-Nick/twi-english-nllb-lora`

That organization repository is used for collaborative access and team-facing
updates while this personal repository remains the original author copy.

## Model Details

- **Task:** Twi to English translation
- **Base model:** `facebook/nllb-200-distilled-600M`
- **Adapter type:** LoRA / PEFT
- **Source language tag:** `twi_Latn`
- **Target language tag:** `eng_Latn`
- **Training hardware:** NVIDIA RTX 2060, 6 GB VRAM
- **Trainable parameters:** 2,359,296 of 617,433,088 total parameters

## Training Data

The adapter was fine-tuned on:

- `TWI_ENGLISH_PARALLEL_TEXT`
- 3,888 training examples
- 431 validation examples

The original CSV used `text` as the Twi source column and `label` as the English
target/reference column.

## Evaluation

Best validation checkpoint: `checkpoint-900`

| Metric | Value | Interpretation |
|---|---:|---|
| BLEU | 27.18 | Decent/useful translation quality for a low-resource language pair. |
| chrF | 48.36 | Good character-level similarity; useful for spelling/morphology variation. |
| Eval loss | 1.5420 | Lower is better; improved from 1.9584 at step 100. |

Translation does not use ordinary classification accuracy because many different
English translations can be valid for the same Twi sentence.

## Usage

```python
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_id = "mclanorjeff/twi-english-nllb-lora"
source_lang = "twi_Latn"
target_lang = "eng_Latn"

def normalize_twi_keyboard_text(text: str) -> str:
    return text.replace("C", "Ɔ").replace("c", "ɔ").replace("3", "ɛ")

config = PeftConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=source_lang, tgt_lang=target_lang)
base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

text = normalize_twi_keyboard_text("3he na wowc? M'ani gye w'as3m ho")
inputs = tokenizer(text, return_tensors="pt").to(device)
forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

with torch.no_grad():
    output = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=192,
        num_beams=5,
        no_repeat_ngram_size=3,
    )

print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
```

## Sample Outputs

| Twi input | Model output |
|---|---|
| `me dc wo` | `i love you` |
| `3he na wowc?` | `where are you?` |
| `3he na wowc? M'ani gye w'as3m ho` | `where are you? I love your story` |
| `Yei nti, ama nnipa pii ani agye ho pa ara.` | `For this reason, it has made it very popular among many people.` |

## Limitations

This is a useful fine-tuned model, not a perfect translator. It may struggle with
slang, idioms, informal Twi, spelling variation, names, or sentences far outside
the training distribution. Human review is recommended for high-stakes use.
