import os
import torch
import random
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = "F:/twi_translation_model"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")

# Data Chunking Settings
START_ROW = 0           # Change this when resuming (e.g., set to 500000 for the 2nd chunk)
NUM_ROWS = 500000       # 500k row sprint

# Languages
TWI_LANG = "aka_GH"     # NLLB code for Akan/Twi
ENG_LANG = "eng_Latn"   # NLLB code for English

BIDIRECTIONAL = True    # If True, randomly swaps Twi->Eng and Eng->Twi

# ==========================================
# 1. SETUP TOKENIZER
# ==========================================
print(f"Loading tokenizer {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=TWI_LANG)

# ==========================================
# 2. LOAD & PREPARE DATASET (Streaming Chunk)
# ==========================================
print("Loading dataset in streaming mode...")
# We use the huggingface hub name, it will use your local cache automatically.
dataset = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences", 
    split="train",
    streaming=True
)

# Skip to our starting row and take the chunk size
chunked_dataset = dataset.skip(START_ROW).take(NUM_ROWS)

def preprocess_function(examples):
    # Determine direction dynamically if bidirectional
    if BIDIRECTIONAL and random.random() > 0.5:
        # Eng -> Twi
        source_lang, target_lang = ENG_LANG, TWI_LANG
        inputs = examples["english"]
        targets = examples["twi"]
    else:
        # Twi -> Eng
        source_lang, target_lang = TWI_LANG, ENG_LANG
        inputs = examples["twi"]
        targets = examples["english"]
        
    tokenizer.src_lang = source_lang
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    
    return model_inputs

print("Preprocessing data chunk...")
# Convert iterable dataset to a list for standard Trainer usage 
# (Since it's only 500k, it will fit in RAM easily after tokenization)
processed_dataset_list = []
for ex in chunked_dataset:
    # process single example
    # Note: normally batched is faster, but streaming iterable datasets handle single map easily
    processed = preprocess_function({k: [v] for k,v in ex.items()})
    processed = {k: v[0] for k,v in processed.items()}
    processed_dataset_list.append(processed)

# ==========================================
# 3. LOAD MODEL IN 4-BIT
# ==========================================
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" # Automatically uses GPU
)

model = prepare_model_for_kbit_training(model)

# ==========================================
# 4. CONFIGURE LoRA OR RESUME
# ==========================================
if os.path.exists(ADAPTER_DIR):
    print(f"Resuming from existing adapter found at {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=True)
else:
    print("Initializing new LoRA adapter...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config)

model.print_trainable_parameters()

# ==========================================
# 5. TRAINING
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        # VERY important for 6GB VRAM
    gradient_accumulation_steps=16,       # Simulate batch size of 16
    learning_rate=2e-4,
    weight_decay=0.01,
    save_strategy="steps",                # Save during training, not just at the end
    save_steps=500,                       # Save every 500 steps (approx every 1.5 hours)
    save_total_limit=10,                  # Keep more checkpoints for the research paper graphs
    logging_steps=10,
    max_steps=len(processed_dataset_list) // 16, # Calculate steps based on batch accumulation
    fp16=True,                            # Fast precision
    optim="paged_adamw_32bit",            # Memory efficient optimizer
    report_to="wandb"                     # Enable Weights & Biases logging
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset_list,
    data_collator=data_collator,
)

print("Starting training sprint (Resuming from latest checkpoint if available)...")
trainer.train(resume_from_checkpoint=True)

# ==========================================
# 6. SAVE ADAPTER
# ==========================================
print(f"Saving new adapter to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print("Done!")
