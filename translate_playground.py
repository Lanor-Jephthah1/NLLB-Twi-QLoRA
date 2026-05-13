import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHECKPOINT_DIR = "F:/twi_translation_model/checkpoint-7500"

# Languages for English -> Twi
SRC_LANG = "eng_Latn"   # English
TGT_LANG = "aka_Latn"     # Official NLLB code for Akan/Twi

# ==========================================
# 1. LOAD MODEL
# ==========================================
print(f"Loading Model for English -> Twi Playground...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)

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

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
model.eval()

def translate(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_length=128
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# ==========================================
# 2. INTERACTIVE LOOP
# ==========================================
print("\n" + "="*50)
print("🌍 TWI TRANSLATION PLAYGROUND (English -> Twi)")
print("="*50)
print("Type an English sentence and press Enter.")
print("Type 'quit' to exit.")

while True:
    user_input = input("\nEnglish: ")
    if user_input.lower() == 'quit':
        break
        
    twi_result = translate(user_input)
    print(f"Twi:     {twi_result}")
