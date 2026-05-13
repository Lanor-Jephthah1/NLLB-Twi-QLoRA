import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import copy

MODEL_NAME = "facebook/nllb-200-distilled-600M"
PHASE1_DIR = "F:/twi_translation_model/checkpoint-12000"
PHASE2_DIR = "F:/twi_translation_model_human/checkpoint-1355"

SRC_LANG = "aka_GH"

# Alpha controls how much Phase 1 weight to retain.
# 1.0 = pure Phase 1, 0.0 = pure Phase 2.
ALPHAS = [0.3, 0.5, 0.7]

SKIP_ROWS = 500000
EVAL_ROWS = 500

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading test set ({EVAL_ROWS} synthetic sentences from rows {SKIP_ROWS}+)...")
raw = load_dataset(
    "ghananlpcommunity/pristine-twi-english-parallel-sentences",
    split="train",
    streaming=True
)
test_slice = list(raw.skip(SKIP_ROWS).take(EVAL_ROWS))
twi_sentences = [ex["twi"] for ex in test_slice]
references = [ex["english"] for ex in test_slice]


def evaluate(model):
    model.eval()
    hypotheses = []
    for sentence in tqdm(twi_sentences, leave=False):
        inputs = tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=128,
                num_beams=4
            )
        hypotheses.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    bleu = BLEU().corpus_score(hypotheses, [references])
    chrf = CHRF(word_order=2).corpus_score(hypotheses, [references])
    return bleu, chrf


def load_fresh_base():
    base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    return prepare_model_for_kbit_training(base)


# Extract LoRA state dicts from both adapters
print(f"Extracting Phase 1 LoRA weights from {PHASE1_DIR}...")
base1 = load_fresh_base()
model_p1 = PeftModel.from_pretrained(base1, PHASE1_DIR, is_trainable=False)
p1_lora = {k: v.clone().cpu() for k, v in model_p1.state_dict().items() if "lora" in k}
del model_p1, base1
torch.cuda.empty_cache()

print(f"Extracting Phase 2 LoRA weights from {PHASE2_DIR}...")
base2 = load_fresh_base()
model_p2 = PeftModel.from_pretrained(base2, PHASE2_DIR, is_trainable=False)
p2_lora = {k: v.clone().cpu() for k, v in model_p2.state_dict().items() if "lora" in k}
del base2
torch.cuda.empty_cache()

print("\n--- Merge Results (500 synthetic test sentences) ---")
print(f"  Reference scores:")
print(f"  Phase 1 only (alpha=1.0) -> BLEU: 43.37 | chrF++: 63.16")
print(f"  Phase 2 only (alpha=0.0) -> BLEU: 41.99 | chrF++: 61.21")
print(f"  ---")

best_bleu = 0
best_alpha = None

for alpha in ALPHAS:
    print(f"\n  Evaluating alpha={alpha} ({int(alpha*100)}% Phase 1 / {int((1-alpha)*100)}% Phase 2)...")

    # Interpolate LoRA weights
    merged_lora = {k: alpha * p1_lora[k].cuda() + (1 - alpha) * p2_lora[k].cuda() for k in p1_lora}

    # Load Phase 2 model and overwrite its LoRA weights with the merged ones
    base = load_fresh_base()
    merged_model = PeftModel.from_pretrained(base, PHASE2_DIR, is_trainable=False)
    current_state = merged_model.state_dict()
    for k in merged_lora:
        current_state[k] = merged_lora[k]
    merged_model.load_state_dict(current_state)

    bleu, chrf = evaluate(merged_model)
    print(f"  BLEU: {bleu}  |  chrF++: {chrf}")

    if float(str(bleu).split()[0]) > best_bleu:
        best_bleu = float(str(bleu).split()[0])
        best_alpha = alpha

    del merged_model, base, merged_lora
    torch.cuda.empty_cache()

print(f"\n--- Best merge: alpha={best_alpha} with BLEU {best_bleu:.2f} ---")
print(f"    To use this model, rerun with only alpha={best_alpha} and save the merged adapter.")
