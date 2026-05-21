from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - only needed for LoRA checkpoints
    PeftConfig = None
    PeftModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick translation using a fine-tuned checkpoint or base model."
    )
    parser.add_argument("--model-path", required=True, help="Checkpoint directory or model name.")
    parser.add_argument("--source-lang", required=True, help="Tokenizer source language tag.")
    parser.add_argument("--target-lang", required=True, help="Tokenizer target language tag.")
    parser.add_argument("--text", required=True, help="Text to translate.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--disable-twi-keyboard-normalization",
        action="store_true",
        help="Do not convert common Twi keyboard substitutions like c -> ɔ and 3 -> ɛ.",
    )
    return parser.parse_args()


def normalize_twi_keyboard_text(text: str) -> str:
    return text.replace("C", "Ɔ").replace("c", "ɔ").replace("3", "ɛ")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    model_path = Path(args.model_path)
    adapter_config_path = model_path / "adapter_config.json"

    if adapter_config_path.exists():
        if PeftConfig is None or PeftModel is None:
            raise ImportError("Install peft to load LoRA adapter checkpoints.")
        peft_config = PeftConfig.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            src_lang=args.source_lang,
            tgt_lang=args.target_lang,
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            src_lang=args.source_lang,
            tgt_lang=args.target_lang,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_text = args.text
    if args.source_lang == "twi_Latn" and not args.disable_twi_keyboard_normalization:
        input_text = normalize_twi_keyboard_text(input_text)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and args.target_lang in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[args.target_lang]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
