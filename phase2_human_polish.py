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
from peft import PeftModel, prepare_model_for_kbit_training

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHECKPOINT_DIR = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/checkpoint-12000"
OUTPUT_DIR = "F:/twi_translation_model_human"
CSV_PATH = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/train.csv"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df[['text', 'label']].rename(columns={'text': 'twi', 'label': 'english'})
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

def preprocess_function(examples):
    inputs = examples["twi"]
    targets = examples["english"]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print(f"Loading base model and Phase 1 adapter from {CHECKPOINT_DIR}...")
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

# prepare_model_for_kbit_training must run before loading the adapter,
# otherwise it will re-freeze the LoRA weights and zero out all gradients.
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR, is_trainable=True)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    # Learning rate is reduced 4x relative to Phase 1 to avoid overwriting
    # the syntactic structure the model learned from the synthetic corpus.
    learning_rate=5e-5,
    num_train_epochs=5,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    predict_with_generate=True,
    # fp16 disabled: causes a GradScaler assertion error with gradient accumulation
    # at batch_size=1 on this hardware configuration.
    fp16=False,
    push_to_hub=False,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting Phase 2 training...")
trainer.train()

print(f"Training complete. Model saved to {OUTPUT_DIR}")
