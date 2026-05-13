import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Point this to whichever checkpoint you want to evaluate.
ADAPTER_DIR = "F:/twi_translation_model_human"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

# We evaluate on 500 rows drawn from the END of the Pristine dataset.
# These rows were never used during Phase 1 (which trained on rows 0-192k)
# and were never part of the human-curated train.csv used in Phase 2.
# This makes the evaluation directly comparable to the Phase 1 checkpoint scores.
EVAL_ROWS = 500
SKIP_ROWS = 500000

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

# Load a clean, unseen test slice from the Pristine dataset.
# Streaming lets us skip to the desired position without downloading the full corpus.
print(f"Loading {EVAL_ROWS} unseen test sentences from GhanaNLP Pristine dataset...")
raw_dataset = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train",
    streaming=True
)
test_slice = list(raw_dataset.skip(SKIP_ROWS).take(EVAL_ROWS))

twi_sentences = [ex["twi"] for ex in test_slice]
reference_translations = [ex["english"] for ex in test_slice]

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

# sacrebleu expects references as a list of lists
references = [reference_translations]

bleu = BLEU()
chrf = CHRF(word_order=2)

bleu_score = bleu.corpus_score(hypotheses, references)
chrf_score = chrf.corpus_score(hypotheses, references)

print("\n--- Evaluation Results (Phase 2: Human-Polished Model) ---")
print(f"  Test set:  {EVAL_ROWS} unseen sentences from GhanaNLP Pristine (rows {SKIP_ROWS}–{SKIP_ROWS + EVAL_ROWS})")
print(f"  BLEU:      {bleu_score}")
print(f"  chrF++:    {chrf_score}")
print("-----------------------------------------------------------")
print("\n  Phase 1 reference scores (same test methodology):")
print("  Checkpoint 3,500  -> BLEU: 42.87 | chrF++: 62.22")
print("  Checkpoint 7,500  -> BLEU: 43.42 | chrF++: 63.05")
print("  Checkpoint 12,000 -> BLEU: 43.37 | chrF++: 63.16")
print("-----------------------------------------------------------")

print("\nSample translations:")
for i in range(min(5, len(hypotheses))):
    print(f"\n  Twi:       {twi_sentences[i]}")
    print(f"  Reference: {reference_translations[i]}")
    print(f"  Model:     {hypotheses[i]}")
