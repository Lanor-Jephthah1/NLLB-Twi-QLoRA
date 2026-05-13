import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "facebook/nllb-200-distilled-600M"
CHECKPOINT_DIR = "F:/twi_translation_model/checkpoint-12000"
SRC_TEXT = "Nkɔsoɔ a aba abisadeɛ nyansahu mu no resesɛ kwan a nipa fa so ne wɔn ho di nkitaho na wɔyɛ adwuma wɔ wiase baabiara."

print("Loading model for visualization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="aka_GH")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# attn_implementation="eager" is required to access raw cross-attention tensors.
# The default SDPA implementation fuses the attention computation and does not
# expose intermediate weights, making extraction impossible.
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
model.eval()

print(f"Running inference on: {SRC_TEXT}")
inputs = tokenizer(SRC_TEXT, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
    max_length=128,
    output_attentions=True,
    return_dict_in_generate=True
)

translated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(f"Translation: {translated_text}")

# cross_attentions shape: (generated_tokens, layers, batch, heads, 1, src_len)
# We take the last decoder layer and average across attention heads to get one
# scalar weight per source token per generated target token.
all_attentions = []
for token_idx in range(len(outputs.cross_attentions)):
    layer_attn = outputs.cross_attentions[token_idx][-1][0]
    avg_heads = layer_attn.mean(dim=0)
    all_attentions.append(avg_heads)

full_attention_matrix = torch.cat(all_attentions, dim=0).cpu().detach().numpy()

src_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
tgt_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])[1:]

plt.figure(figsize=(15, 12))
sns.heatmap(
    full_attention_matrix,
    xticklabels=src_tokens,
    yticklabels=tgt_tokens,
    cmap='viridis',
    annot=False
)
plt.title(f"Cross-Attention Map: Twi to English (Step 12000)\n'{SRC_TEXT}'", fontsize=14)
plt.xlabel("Source Tokens (Twi)", fontsize=12)
plt.ylabel("Target Tokens (English)", fontsize=12)

save_path = "C:/Users/McLanor Jeff/.gemini/antigravity/scratch/attention_map.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Attention map saved to: {save_path}")
plt.show()
