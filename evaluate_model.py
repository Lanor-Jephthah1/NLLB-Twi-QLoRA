import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_DIR = "F:/twi_translation_model_human/checkpoint-1355"

SRC_LANG = "aka_GH"
TGT_LANG = "eng_Latn"

# Same 500 synthetic test sentences used to evaluate all Phase 1 checkpoints.
# These rows were never used in Phase 1 training (which covered rows 0-192k).
SKIP_ROWS = 500000
EVAL_ROWS = 500

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

print(f"Loading {EVAL_ROWS} test sentences from GhanaNLP Pristine (rows {SKIP_ROWS}+)...")
raw_dataset = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train",
    streaming=True
)
test_slice = list(raw_dataset.skip(SKIP_ROWS).take(EVAL_ROWS))
twi_sentences = [ex["twi"] for ex in test_slice]
references = [ex["english"] for ex in test_slice]

print("Running inference...")
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
    hypotheses.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

bleu_score = BLEU().corpus_score(hypotheses, [references])
chrf_score = CHRF(word_order=2).corpus_score(hypotheses, [references])

print("\n--- Results ---")
print(f"  Phase 2 (checkpoint-1355) — 500 synthetic test sentences")
print(f"  BLEU:   {bleu_score}")
print(f"  chrF++: {chrf_score}")
print(f"\n  Phase 1 reference (same test set):")
print(f"  Checkpoint 12,000 -> BLEU: 43.37 | chrF++: 63.16")
print("----------------")

print("\nSample translations:")
for i in range(min(5, len(hypotheses))):
    print(f"\n  Twi:       {twi_sentences[i]}")
    print(f"  Reference: {references[i]}")
    print(f"  Model:     {hypotheses[i]}")
