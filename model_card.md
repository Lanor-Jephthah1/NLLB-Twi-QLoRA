---
license: mit
base_model: facebook/nllb-200-distilled-600M
tags:
- nllb
- translation
- twi
- akan
- qlora
- peft
datasets:
- ghananlpcommunity/pristine-twi-english-parallel-sentences
metrics:
- bleu
- chrf
language:
- ak
- en
library_name: peft
---

# NLLB-Twi Human-Aligned Translator (Phase 2)

This repository contains a QLoRA adapter for the NLLB-200 (600M) model, specifically fine-tuned for high-fidelity Twi-to-English translation. This model is the result of a two-stage alignment framework designed to achieve state-of-the-art performance on low-resource hardware.

## Performance Metrics
| Metric | Score |
|---|---|
| BLEU | 41.99 |
| chrF++ | 61.21 |

*Note: Evaluation was conducted on a held-out set of 500 sentences from the GhanaNLP Pristine corpus.*

## Training Methodology
The model was developed using a two-stage curriculum to balance structural accuracy with conversational naturalness:

1. **Phase 1: Synthetic Scaling**: Initial training on 192,000 parallel sentences from the GhanaNLP Pristine corpus. This phase established the foundational morphological and syntactic patterns of Twi.
2. **Phase 2: Human Alignment**: Refinement on 4,331 human-verified sentence pairs. This stage addressed the "robotic" stylistic artifacts typical of synthetic data, aligning the model with natural Twi phrasing.

### Technical Specifications
- **Hardware**: NVIDIA RTX 2060 (6GB VRAM).
- **Quantization**: 4-bit NormalFloat (NF4) via bitsandbytes.
- **Methodology**: QLoRA (Rank 16, Alpha 32).
- **Optimizer**: Paged AdamW (8-bit).
- **Learning Rate**: 2e-4 (Phase 1) reduced to 5e-5 (Phase 2) for conservative alignment.

## Usage
The model can be loaded directly using the Transformers library. The base NLLB model weights will be inferred and loaded automatically from the adapter configuration.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_id = "mclanorjeff/NLLB-Twi-Human-Aligned"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="aka_GH")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Inference
text = "Meresua Twi kasa kyerɛ wo."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        max_length=128,
        num_beams=5
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Intended Use
This model is intended for research and development in African NLP. It is optimized for Twi-to-English translation across various domains, including conversational and formal text.

## Acknowledgements
We acknowledge the **GhanaNLP community** for providing the foundational datasets that made this research possible.
