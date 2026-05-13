import os
import torch
import random
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, prepare_model_for_kbit_training

MODEL_NAME = "facebook/nllb-200-distilled-600M"
PHASE1_CHECKPOINT = "F:/twi_translation_model/checkpoint-12000"
OUTPUT_DIR = "F:/twi_translation_model_human_v2"
HUMAN_CSV = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/train.csv"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

# Synthetic chunk: 20,000 sentences from a region not seen in Phase 1.
# Phase 1 covered rows 0-192,000. We start well beyond that.
SYNTHETIC_SKIP = 200000
SYNTHETIC_ROWS = 20000

# Human sentences are repeated 5x so they carry comparable weight
# to the 20k synthetic sentences in the mixed dataset.
HUMAN_REPEAT = 5

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)

# --- Build combined dataset ---

print(f"Loading {SYNTHETIC_ROWS} synthetic sentences (rows {SYNTHETIC_SKIP}+)...")
raw = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train",
    streaming=True
)
synthetic_slice = list(raw.skip(SYNTHETIC_SKIP).take(SYNTHETIC_ROWS))
synthetic_pairs = [{"twi": ex["twi"], "english": ex["english"]} for ex in synthetic_slice]

print(f"Loading human sentences from {HUMAN_CSV} (repeated {HUMAN_REPEAT}x)...")
df = pd.read_csv(HUMAN_CSV)
human_pairs = []
for _ in range(HUMAN_REPEAT):
    for _, row in df.iterrows():
        human_pairs.append({"twi": row["text"], "english": row["label"]})

combined = synthetic_pairs + human_pairs
random.shuffle(combined)
print(f"Combined dataset: {len(combined)} samples ({SYNTHETIC_ROWS} synthetic + {len(df) * HUMAN_REPEAT} human exposures)")

def preprocess(examples):
    tokenizer.src_lang = SRC_LANG
    return tokenizer(
        examples["twi"],
        text_target=examples["english"],
        max_length=128,
        truncation=True
    )

hf_dataset = Dataset.from_list(combined)
tokenized = hf_dataset.map(preprocess, batched=True, remove_columns=hf_dataset.column_names)

# --- Load model ---

print(f"Loading base model and Phase 1 adapter from {PHASE1_CHECKPOINT}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# prepare_model_for_kbit_training must run before PeftModel.from_pretrained.
# If the order is reversed, the LoRA weights are re-frozen and grad_norm goes to zero.
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, PHASE1_CHECKPOINT, is_trainable=True)
model.print_trainable_parameters()

# --- Train ---

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    # Lower LR than Phase 1 to preserve the synthetic knowledge while absorbing human patterns
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("Starting Phase 2 training on combined synthetic + human dataset...")
trainer.train()

print(f"Training complete. Model saved to {OUTPUT_DIR}")
