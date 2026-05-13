import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import os

MODEL_NAME = "facebook/nllb-200-distilled-600M"
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

# Load Phase 1 LoRA weights
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

# Filter and normalize Phase 1 keys (remove .weight for matching if needed, but here we just need them as-is)
p1_weights = {k: v for k, v in p1_raw.items() if "lora_A" in k or "lora_B" in k}

# Load Phase 2 model
print(f"Loading Phase 2 model from {PHASE2_DIR}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
base = prepare_model_for_kbit_training(base)
model = PeftModel.from_pretrained(base, PHASE2_DIR, is_trainable=True)

# Snapshot Phase 2 weights
p2_weights = {k: v.clone().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}

# Map keys by stripping the '.default.' suffix
mapping = {}
for k2 in p2_weights.keys():
    k1_guess = k2.replace(".default.", ".")
    if k1_guess in p1_weights:
        mapping[k2] = k1_guess

print(f"  Mapped {len(mapping)} LoRA tensors between Phase 1 and Phase 2.")

if len(mapping) == 0:
    print("  ERROR: Still no matching keys. Naming logic must be different.")
    print("  P1 example:", list(p1_weights.keys())[0])
    print("  P2 example:", list(p2_weights.keys())[0])
    import sys; sys.exit(1)

def apply_merge(alpha):
    state = model.state_dict()
    for k2, k1 in mapping.items():
        state[k2] = alpha * p1_weights[k1].to(model.device) + (1 - alpha) * p2_weights[k2].to(model.device)
    model.load_state_dict(state)

def restore_phase2():
    state = model.state_dict()
    for k2 in p2_weights:
        state[k2] = p2_weights[k2].to(model.device)
    model.load_state_dict(state)

print("\n--- Merge Results (500 synthetic test sentences) ---")
print("  Phase 1 reference -> BLEU: 43.37")
print("  Phase 2 reference -> BLEU: 41.99")
print("  ---")

best_bleu = 0.0
best_alpha = None

for alpha in ALPHAS:
    print(f"\n  alpha={alpha}...")
    apply_merge(alpha)
    bleu, chrf = evaluate(model)
    bleu_val = float(str(bleu).split()[0])
    print(f"  BLEU: {bleu}  |  chrF++: {chrf}")
    if bleu_val > best_bleu:
        best_bleu = bleu_val
        best_alpha = alpha
    restore_phase2()

print(f"\n--- Best: alpha={best_alpha}  BLEU={best_bleu:.2f} ---")
