import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION ---
ADAPTER_ID = "mclanorjeff/NLLB-Twi-Human-Aligned" 
BASE_MODEL = "facebook/nllb-200-distilled-600M"

print("Loading model and tokenizer... this may take a few minutes.")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang="aka_GH")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, ADAPTER_ID)
model.eval()

def translate(text):
    if not text.strip():
        return ""
    
    # Automatic character mapping for English keyboards
    text = text.replace('3', 'ɛ').replace('c', 'ɔ').replace('C', 'Ɔ')
    
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128,
            num_beams=5
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 900px !important;
    margin: auto;
}
.header-box {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Column(elem_classes="header-box"):
        gr.Markdown("# 🇬🇭 NLLB-Twi Translator (Phase 2)")
        gr.Markdown("### Professional Human-Aligned Machine Translation")
        gr.Markdown("This model uses a two-stage QLoRA approach, combining 192k synthetic sentences with human-verified polish.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Enter Twi Text", 
                placeholder="Type here... (Tip: Use '3' for 'ɛ' and 'c' for 'ɔ')",
                lines=8
            )
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="English Translation", lines=8, interactive=False)

    with gr.Row():
        gr.Column(scale=1) # Spacer
        translate_btn = gr.Button("Translate to English", variant="primary", scale=1)
        gr.Column(scale=1) # Spacer

    gr.Examples(
        examples=[
            ["Meresua Twi kasa kyerɛ wo."],
            ["S3 obi kcc sukuu a, 3s3 s3 obc mbcden."],
            ["Mepa wo kyɛw, kyerɛ me kwan a ɛkɔ sukuu hɔ."],
            ["Ghana yɛ ɔman a ɛyɛ fɛ paa."],
        ],
        inputs=input_text
    )

    translate_btn.click(translate, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
