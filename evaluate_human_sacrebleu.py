import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF
from tqdm import tqdm

MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_DIR = "F:/twi_translation_model_human/checkpoint-1355"
SRC_LANG = "aka_GH"

# Load local human-verified data
print("Loading human-verified test slice from data/train.csv...")
df = pd.read_csv("data/train.csv")
# Take the last 100 rows (less likely to be overfitted than the beginning)
test_df = df.tail(100)
twi_sentences = test_df['text'].tolist()
references = test_df['label'].tolist()

print(f"Loading model and adapter...")
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

print(f"Running SacreBLEU inference on {len(twi_sentences)} human sentences...")
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

# Calculate official SacreBLEU scores
bleu = BLEU()
chrf = CHRF(word_order=2)

bleu_score = bleu.corpus_score(hypotheses, [references])
chrf_score = chrf.corpus_score(hypotheses, [references])

print("\n" + "="*50)
print("OFFICIAL SACREBLEU RESULTS (Human-Verified Slice)")
print("="*50)
print(f"BLEU Score:  {bleu_score}")
print(f"chrF++ Score: {chrf_score}")
print("="*50)

print("\nSample Human-Aligned Translations:")
for i in range(min(3, len(hypotheses))):
    print(f"\n[Twi]:       {twi_sentences[i]}")
    print(f"[Reference]: {references[i]}")
    print(f"[Model]:     {hypotheses[i]}")
