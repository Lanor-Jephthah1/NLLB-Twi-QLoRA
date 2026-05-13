import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-distilled-600M"
# checkpoint-1355 is the final epoch (epoch 5/5) of Phase 2 human-polished training
ADAPTER_DIR = "F:/twi_translation_model_human/checkpoint-1355"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

# Synthetic test slice: rows beyond the 500k training region, never seen in Phase 1
SYNTHETIC_SKIP = 500000
SYNTHETIC_EVAL_ROWS = 300

# Human test slice: the last 200 rows of train.csv, held out from Phase 2 training.
# Phase 2 trained on rows 0-4131. Rows 4131-4331 are reserved for evaluation only.
HUMAN_CSV = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/train.csv"
HUMAN_EVAL_ROWS = 200

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

# --- Load synthetic test slice ---
print(f"Loading {SYNTHETIC_EVAL_ROWS} unseen synthetic sentences (rows {SYNTHETIC_SKIP}+)...")
raw_dataset = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train",
    streaming=True
)
synthetic_slice = list(raw_dataset.skip(SYNTHETIC_SKIP).take(SYNTHETIC_EVAL_ROWS))
synthetic_twi = [ex["twi"] for ex in synthetic_slice]
synthetic_refs = [ex["english"] for ex in synthetic_slice]

# --- Load human test slice ---
print(f"Loading {HUMAN_EVAL_ROWS} held-out human sentences from train.csv...")
df = pd.read_csv(HUMAN_CSV)
human_slice = df.tail(HUMAN_EVAL_ROWS)
human_twi = human_slice['text'].tolist()
human_refs = human_slice['label'].tolist()

# --- Combine both sources into one test set ---
all_twi = synthetic_twi + human_twi
all_refs = synthetic_refs + human_refs
print(f"Combined test set: {len(all_twi)} sentences ({SYNTHETIC_EVAL_ROWS} synthetic + {HUMAN_EVAL_ROWS} human)")

# --- Run inference ---
print("Running inference...")
hypotheses = []

for sentence in tqdm(all_twi):
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

references = [all_refs]

bleu = BLEU()
chrf = CHRF(word_order=2)

bleu_score  = bleu.corpus_score(hypotheses, references)
chrf_score  = chrf.corpus_score(hypotheses, references)

# Also score each domain separately for breakdown analysis
synthetic_hyps = hypotheses[:SYNTHETIC_EVAL_ROWS]
human_hyps     = hypotheses[SYNTHETIC_EVAL_ROWS:]

bleu_synthetic = bleu.corpus_score(synthetic_hyps, [synthetic_refs])
bleu_human     = bleu.corpus_score(human_hyps,     [human_refs])
chrf_synthetic = chrf.corpus_score(synthetic_hyps, [synthetic_refs])
chrf_human     = chrf.corpus_score(human_hyps,     [human_refs])

print("\n--- Evaluation Results (Phase 2: Human-Polished Model) ---")
print(f"  Combined ({len(all_twi)} sentences):")
print(f"    BLEU:   {bleu_score}")
print(f"    chrF++: {chrf_score}")
print(f"\n  Synthetic domain ({SYNTHETIC_EVAL_ROWS} sentences):")
print(f"    BLEU:   {bleu_synthetic}")
print(f"    chrF++: {chrf_synthetic}")
print(f"\n  Human domain ({HUMAN_EVAL_ROWS} sentences):")
print(f"    BLEU:   {bleu_human}")
print(f"    chrF++: {chrf_human}")
print("-----------------------------------------------------------")
print("\n  Phase 1 reference scores (500 synthetic sentences):")
print("  Checkpoint 3,500  -> BLEU: 42.87 | chrF++: 62.22")
print("  Checkpoint 7,500  -> BLEU: 43.42 | chrF++: 63.05")
print("  Checkpoint 12,000 -> BLEU: 43.37 | chrF++: 63.16")
print("-----------------------------------------------------------")

print("\nSample translations:")
for i in range(min(5, len(hypotheses))):
    print(f"\n  Twi:       {all_twi[i]}")
    print(f"  Reference: {all_refs[i]}")
    print(f"  Model:     {hypotheses[i]}")
