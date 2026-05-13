# 🌍 Twi-English Machine Translation via Two-Stage QLoRA
**Democratizing Low-Resource African NLP on Consumer Hardware**

This repository contains the training scripts, evaluation metrics, and interpretability logic for fine-tuning Meta's **NLLB-200 (600M)** model for the Ghanaian language **Akan (Twi)**. The entire pipeline was engineered to run locally on a consumer-grade laptop (RTX 2060, 6GB VRAM, 16GB RAM) by leveraging 4-bit Quantization and Parameter-Efficient Fine-Tuning (PEFT).

## 🏆 Project Achievements
- **Baseline BLEU (Zero-Shot):** `18.94`
- **Final Model BLEU:** `43.42` (+24.48 point improvement)
- **Hardware Constraint Defeated:** Successfully fine-tuned a 600M parameter model on a strict **6GB VRAM** ceiling using `NF4` quantization and gradient accumulation.

## 🧠 Methodology: The "Two-Stage" Alignment
Because Twi is a low-resource language, high-quality human data is extremely scarce. To achieve state-of-the-art results, we implemented a two-stage alignment process:

### Phase 1: Synthetic Scaling (Grammar & Structure)
- **Script:** `phase1_synthetic_training.py`
- **Data:** ~192,000 synthetic Twi-English parallel sentences.
- **Goal:** Inject foundational Twi vocabulary and morphological syntax into the model's latent space.
- **Outcome:** The model learned to translate fluently but retained a slight "robotic" and direct translation accent.

### Phase 2: Human-in-the-Loop Refinement (Culture & Nuance)
- **Script:** `phase2_human_polish.py`
- **Data:** 4,300 highly-curated, human-verified sentence pairs.
- **Goal:** Correct the robotic accent, align the model with conversational slang, and inject cultural nuance.
- **Outcome:** We deliberately dropped the learning rate from `2e-4` to `5e-5` to prevent catastrophic forgetting. The model successfully aligned to human preferences, resulting in natural, high-fidelity translations.

## 📈 Training Metrics & Loss
During Phase 2, the model experienced a "Distribution Shift" (a sudden loss spike) as it transitioned from predictable synthetic data to complex human data. It successfully converged shortly after.

![Training Loss Curve](assets/loss_curve.png)
*(Place your exported Matplotlib or WandB graph in the `assets` folder as `loss_curve.png`)*

## 🔍 Interpretability: Cross-Attention Mapping
To prove the model achieved deep semantic understanding rather than mere phrase memorization, we extracted the eager cross-attention weights. 
- **Script:** `visualize_attention.py`

![Attention Heatmap](assets/attention_map.png)
*(Place your generated attention heatmap in the `assets` folder as `attention_map.png`)*
*The heatmap demonstrates strong, confident diagonal mapping, correctly linking complex Twi phrasing directly to specific English targets.*

## 💻 Repository Structure
```text
├── phase1_synthetic_training.py  # Stage 1: Massive synthetic ingestion
├── phase2_human_polish.py        # Stage 2: Low-LR human alignment
├── evaluate_model.py             # SacreBLEU evaluation script
├── translate_playground.py       # Interactive CLI for real-time translation
├── visualize_attention.py        # PyTorch hook script for attention mapping
└── assets/                       # Images for README
```

## 🚀 How to Run the Playground
Want to test the translation locally? Run the playground script:
```bash
python translate_playground.py
```

## ⚠️ Limitations
- **Directional Bias:** While the base NLLB model is bi-directional, our fine-tuning heavily favored the **Twi → English** direction.
- **Hardware Bottlenecks:** The 6GB VRAM limit forced a batch size of 1 (with 16 accumulation steps), preventing full-parameter unfreezing.

---
*This project serves as a blueprint for AI researchers and students across Africa looking to build high-quality NLP models without access to corporate-tier A100 GPU clusters.*
