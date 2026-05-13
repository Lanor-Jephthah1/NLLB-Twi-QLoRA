import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHECKPOINT_DIR = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/checkpoint-12000" # GRADUATE FROM HERE
OUTPUT_DIR = "F:/twi_translation_model_human"               # NEW FOLDER FOR HUMAN MODEL
CSV_PATH = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/train.csv"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

# ==========================================
# 1. LOAD LOCAL HUMAN DATASET
# ==========================================
print(f"Loading local human dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
# Map columns: text -> twi, label -> english
df = df[['text', 'label']].rename(columns={'text': 'twi', 'label': 'english'})

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(df)

# ==========================================
# 2. TOKENIZATION
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

def preprocess_function(examples):
    inputs = examples["twi"]
    targets = examples["english"]
    
    # Modern approach: use text_target parameter directly
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

print("Tokenizing human dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# ==========================================
# 3. LOAD MODEL (RESUME FROM 12000)
# ==========================================
print(f"Loading Model from Checkpoint 12000...")
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

model = prepare_model_for_kbit_training(model)
# Load the LoRA weights from the checkpoint and set them as trainable
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR, is_trainable=True)
model.print_trainable_parameters()

# ==========================================
# 4. TRAINING ARGUMENTS (GENTLE SETTINGS)
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,   # 1 is safest for 6GB VRAM
    gradient_accumulation_steps=16,  # Effective batch size = 16
    learning_rate=5e-5,             # 4x smaller than before (gentle tuning)
    num_train_epochs=5,             # 5 passes over the human data
    logging_steps=10,
    save_strategy="epoch",          # Save at the end of every human epoch
    eval_strategy="no",
    predict_with_generate=True,
    fp16=False,             # Disabled to fix "No inf checks" error
    push_to_hub=False,
    report_to="none",       # Disabled WandB to keep things simple
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ==========================================
# 5. START POLISHING!
# ==========================================
print("\n🚀 Starting Human Polish phase...")
trainer.train()

print(f"\n✅ Training Complete! Human-polished model saved to {OUTPUT_DIR}")
