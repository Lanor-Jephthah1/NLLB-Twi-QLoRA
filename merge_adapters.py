import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import os

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# lora_adapter contains the clean LoRA-only weights saved at the end of Phase 1.
# checkpoint-12000 is a full Trainer checkpoint (includes quantized base weights)
# and cannot be loaded as a raw state dict.
PHASE1_LORA_DIR = "F:/twi_translation_model/lora_adapter"
PHASE2_DIR = "F:/twi_translation_model_human/checkpoint-1355"

SRC_LANG = "aka_GH"
ALPHAS = [0.3, 0.5, 0.7]

SKIP_ROWS = 500000
EVAL_ROWS = 500

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)

print(f"Loading {EVAL_ROWS} synthetic test sentences (rows {SKIP_ROWS}+)...")
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
    return BLEU().corpus_score(hypotheses, [references]), CHRF(word_order=2).corpus_score(hypotheses, [references])


# Load Phase 1 LoRA weights directly from disk to CPU.
# This reads only the small adapter file, not the full quantized model.
print(f"Loading Phase 1 LoRA weights from {PHASE1_LORA_DIR}...")
p1_safe = os.path.join(PHASE1_LORA_DIR, "adapter_model.safetensors")
p1_bin  = os.path.join(PHASE1_LORA_DIR, "adapter_model.bin")

if os.path.exists(p1_safe):
    from safetensors.torch import load_file
    p1_raw = load_file(p1_safe, device="cpu")
elif os.path.exists(p1_bin):
    p1_raw = torch.load(p1_bin, map_location="cpu")
else:
    raise FileNotFoundError(f"No adapter_model file found in {PHASE1_LORA_DIR}")

# Keep only the actual trainable LoRA parameter tensors
p1_weights = {k: v for k, v in p1_raw.items() if "lora_A" in k or "lora_B" in k}
print(f"  Found {len(p1_weights)} LoRA tensors (lora_A + lora_B)")
if len(p1_weights) == 0:
    print("  ERROR: No LoRA keys found. Check that PHASE1_LORA_DIR points to the adapter folder.")
    raise SystemExit

# Load Phase 2 model into VRAM — the only model kept in GPU memory throughout.
print(f"Loading Phase 2 model from {PHASE2_DIR}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
base = prepare_model_for_kbit_training(base)
model = PeftModel.from_pretrained(base, PHASE2_DIR, is_trainable=True)

# Snapshot Phase 2 LoRA weights so we can restore them between alpha evaluations
p2_weights = {k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
print(f"  Found {len(p2_weights)} LoRA tensors in Phase 2 model")

# Verify key overlap
common = set(p1_weights.keys()) & set(p2_weights.keys())
print(f"  Matching keys between Phase 1 and Phase 2: {len(common)}")
if len(common) == 0:
    print("  WARNING: No matching LoRA keys found. Printing both key sets for debugging:")
    print("  Phase 1 keys:", list(p1_weights.keys())[:5])
    print("  Phase 2 keys:", list(p2_weights.keys())[:5])
    raise SystemExit


def apply_merge(alpha):
    state = model.state_dict()
    for k in common:
        state[k] = alpha * p1_weights[k].to(model.device) + (1 - alpha) * p2_weights[k].to(model.device)
    model.load_state_dict(state)


def restore_phase2():
    state = model.state_dict()
    for k in p2_weights:
        state[k] = p2_weights[k].to(model.device)
    model.load_state_dict(state)


print("\n--- Merge Results (500 synthetic test sentences) ---")
print("  Phase 1 only (alpha=1.0) -> BLEU: 43.37 | chrF++: 63.16  [known]")
print("  Phase 2 only (alpha=0.0) -> BLEU: 41.99 | chrF++: 61.21  [known]")
print("  ---")

best_bleu = 0.0
best_alpha = None

for alpha in ALPHAS:
    print(f"\n  alpha={alpha}  ({int(alpha*100)}% Phase 1 / {int((1-alpha)*100)}% Phase 2)...")
    apply_merge(alpha)
    bleu, chrf = evaluate(model)
    bleu_val = float(str(bleu).split()[0])
    print(f"  BLEU: {bleu}  |  chrF++: {chrf}")
    if bleu_val > best_bleu:
        best_bleu = bleu_val
        best_alpha = alpha
    restore_phase2()

print(f"\n--- Best merge: alpha={best_alpha}  BLEU={best_bleu:.2f} ---")
