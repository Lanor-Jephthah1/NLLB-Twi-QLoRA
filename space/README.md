---
title: NLLB Twi-English Translator (Phase 2)
emoji: 🇬🇭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
license: mit
short_description: Professional Human-Aligned Twi-to-English Translator
---

# NLLB-Twi Translator

This Space hosts the Phase 2 Human-Aligned version of the Twi-English translation model. 

## Research Methodology
This model was developed using a two-stage QLoRA fine-tuning process:
1. **Phase 1 (Synthetic)**: 192,000 parallel sentences.
2. **Phase 2 (Human)**: 4,331 human-verified parallel sentences.

Final BLEU Score: **41.99**

## How to use
Type Twi text into the input box and click "Translate". Use the on-screen buttons for Twi-specific characters (ɛ and ɔ).
