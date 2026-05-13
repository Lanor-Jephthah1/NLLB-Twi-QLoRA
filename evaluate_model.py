import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Point this to whichever checkpoint you want to evaluate.
# For the Phase 2 final model, point to the last saved epoch in OUTPUT_DIR.
ADAPTER_DIR = "F:/twi_translation_model_human"

TEST_CSV = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/train.csv"

# Number of rows to evaluate. Using a held-out slice from the end of the dataset.
# Ensure these rows were NOT seen during training if you want a clean test split.
EVAL_ROWS = 500

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

print(f"Loading adapter from {ADAPTER_DIR}...")
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
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

print(f"Loading test set from {TEST_CSV} (last {EVAL_ROWS} rows)...")
df = pd.read_csv(TEST_CSV).tail(EVAL_ROWS)
twi_sentences = df['text'].tolist()
reference_translations = df['label'].tolist()

print(f"Running inference on {EVAL_ROWS} sentences...")
hypotheses = []

for sentence in tqdm(twi_sentences):
    inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=4
        )
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    hypotheses.append(translated)

# sacrebleu expects references as a list of lists (one list per reference set)
references = [reference_translations]

bleu = BLEU()
chrf = CHRF(word_order=2)

bleu_score = bleu.corpus_score(hypotheses, references)
chrf_score = chrf.corpus_score(hypotheses, references)

print("\n--- Evaluation Results ---")
print(f"BLEU:   {bleu_score}")
print(f"chrF++: {chrf_score}")
print("--------------------------")

# Print a few side-by-side examples for qualitative inspection
print("\nSample translations:")
for i in range(min(5, len(hypotheses))):
    print(f"\n  Twi:       {twi_sentences[i]}")
    print(f"  Reference: {reference_translations[i]}")
    print(f"  Model:     {hypotheses[i]}")
