import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm
import os

MODEL_NAME = "facebook/nllb-200-distilled-600M"
PHASE1_DIR = "F:/twi_translation_model/checkpoint-12000"
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


# Load Phase 1 adapter weights directly from disk to CPU — no full model needed.
# LoRA adapter files are only a few MB, so this fits comfortably in RAM.
print(f"Loading Phase 1 LoRA weights from disk (CPU only)...")
p1_bin = os.path.join(PHASE1_DIR, "adapter_model.bin")
p1_safe = os.path.join(PHASE1_DIR, "adapter_model.safetensors")

if os.path.exists(p1_safe):
    from safetensors.torch import load_file
    p1_weights = load_file(p1_safe, device="cpu")
else:
    p1_weights = torch.load(p1_bin, map_location="cpu")

print(f"Phase 1 adapter has {len(p1_weights)} LoRA tensors")

# Load ONE model (Phase 2) into VRAM — this is the only model kept in GPU memory.
print(f"Loading Phase 2 model into VRAM from {PHASE2_DIR}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
base = prepare_model_for_kbit_training(base)
model = PeftModel.from_pretrained(base, PHASE2_DIR, is_trainable=True)

# Save Phase 2 LoRA weights as the baseline to restore between evaluations
p2_weights = {k: v.clone().cpu() for k, v in model.state_dict().items() if "lora" in k}
print(f"Phase 2 adapter has {len(p2_weights)} LoRA tensors")


def apply_merge(alpha):
    state = model.state_dict()
    for k in p2_weights:
        if k in p1_weights:
            state[k] = (alpha * p1_weights[k].to(model.device) + (1 - alpha) * p2_weights[k].to(model.device))
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

print(f"\n--- Best: alpha={best_alpha}  BLEU={best_bleu:.2f} ---")
print(f"    To deploy this blend, rerun apply_merge({best_alpha}) and save the model.")
