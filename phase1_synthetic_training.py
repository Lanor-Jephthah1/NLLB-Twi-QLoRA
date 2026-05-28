import os 
import torch
import random 
from transformers import (
    AutoModelForSeq2SeqLM, # loads the NLLB from huggingface based on its repo name
    AutoTokenizer, 
    Seq2SeqTrainingArguments, # holds all training configs. 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig # used to configure 4/8 bit quantization
) 

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset 

# Paths and model identifiers
MODEL_NAME = "facebook/nllb-200-distilled-600M" 
OUTPUT_DIR = "F:/twi_translation_model"
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter") #  Combines the output directory with a subfolder name to designate where the final trained LoRA adapter weight will be saved

# Increment START_ROW by NUM_ROWS when moving to the next chunk.
START_ROW = 0
NUM_ROWS = 500000 # initial planned limit for data injection to 500k pairs.

TWI_LANG = "aka_GH"
ENG_LANG = "eng_Latn"

# When True, each example is randomly assigned a direction (Twi->Eng or Eng->Twi)
# so the adapter learns both translation directions simultaneously.
BIDIRECTIONAL = True

print(f"Loading tokenizer {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=TWI_LANG)
# takes raw words and turns them into numeric IDs that the neural network can process

print("Loading dataset in streaming mode...")
dataset = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train", #loads training split
    streaming=True #avoids downloading the entire dataset
)

chunked_dataset = dataset.skip(START_ROW).take(NUM_ROWS) #prepares to extract the 500K parallel sentence pairs

# implements bidirectional logic
def preprocess_function(examples):
    if BIDIRECTIONAL and random.random() > 0.5:
        source_lang, target_lang = ENG_LANG, TWI_LANG
        inputs = examples["english"]
        targets = examples["twi"]
    else:
        source_lang, target_lang = TWI_LANG, ENG_LANG
        inputs = examples["twi"]
        targets = examples["english"]

    tokenizer.src_lang = source_lang #current source language of the tokenizer
    # max_length=128 keeps attention matrix size within the 6GB VRAM constraint
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
# restricting sequence length to 128 guarantees the model fits inside the 6gb GPU

print("Preprocessing data chunk...")
# streaming datasets must be converted to a list before passing to Seq2SeqTrainer.
# at 500k rows this fits comfortably in 16GB RAM after tokenization.
processed_dataset_list = []
for ex in chunked_dataset:
    processed = preprocess_function({k: [v] for k, v in ex.items()})
    processed = {k: v[0] for k, v in processed.items()} #formats each sample as a dict 
    processed_dataset_list.append(processed) #appends the tokenized record to the processed dataset list

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, #compresses to 4bits upon loading
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", #normalfloat 4 datatype, preserves maximum accuracy for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16
)
# sets up the Qlora params

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" #automatically places the quantized layers onto the GPU
)


model = prepare_model_for_kbit_training(model)
# Adjusts the model for low-bit training. It enables gradient checkpointing (to save VRAM) and configures the parameters to
# calculate gradients correctly through 4-bit layers.

# Resume from an existing adapter if one exists, otherwise initialise a new one.
if os.path.exists(ADAPTER_DIR):
    print(f"Resuming from adapter at {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=True)
else:
    print("Initialising new LoRA adapter...")
    peft_config = LoraConfig(
        r=16, #rank of the low matrices.
        lora_alpha=32, # a scaling factor that regulates how heavily our adapter updates modify the original base model weights.
        target_modules=["q_proj", "v_proj"],
        # tells LoRA to insert trainable adapters into the Query and Value 
        # projections inside the attention mechanism
        lora_dropout=0.05, #applies a 5% dropout rate to protect against training data overfitting.
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config) # wraps the base model and attaches the configured LoRA layers.

model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR, 
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4, # step size the optimiser uses to adjust weights
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=10,
    logging_steps=10, #logs metrics to the console after every 10 steps
    # steps derived from dataset size divided by effective batch size
    max_steps=len(processed_dataset_list) // 16,
    fp16=True, #utilizes half-precision (16-bit) training, which speeds up processing and cuts GPU memory usage
    optim="paged_adamw_32bit",
    report_to="wandb"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset_list,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train(resume_from_checkpoint=True)

print(f"Saving adapter to {ADAPTER_DIR}...")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
