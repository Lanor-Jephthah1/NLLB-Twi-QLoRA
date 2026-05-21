from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from translation_config import NLLB_RECOMMENDED_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the NLLB distilled model on a processed Ghana translation dataset."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Path to a processed JSONL train split.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        required=True,
        help="Path to a processed JSONL validation split.",
    )
    parser.add_argument(
        "--model-name",
        default=NLLB_RECOMMENDED_MODEL,
        help="Base seq2seq translation model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where checkpoints and the final model will be saved.",
    )
    parser.add_argument("--source-lang", required=True, help="Tokenizer source language tag.")
    parser.add_argument("--target-lang", required=True, help="Tokenizer target language tag.")
    parser.add_argument("--max-source-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--generation-num-beams", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Train memory-efficient LoRA adapter weights instead of full model weights.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated module names to adapt with LoRA.",
    )
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce GPU memory use at the cost of slightly slower training.",
    )
    parser.add_argument(
        "--disable-fp16",
        action="store_true",
        help="Turn off fp16 mixed precision even if CUDA is available.",
    )
    return parser.parse_args()


def validate_language_tags(tokenizer: AutoTokenizer, source_lang: str, target_lang: str) -> None:
    lang_codes = getattr(tokenizer, "lang_code_to_id", None)
    if not lang_codes:
        return

    missing = [lang for lang in (source_lang, target_lang) if lang not in lang_codes]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"The tokenizer does not recognize these language tags: {missing_str}. "
            "For NLLB in this repo, start with eng_Latn, twi_Latn, or ewe_Latn."
        )


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.validation_file),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
    )
    validate_language_tags(tokenizer, args.source_lang, args.target_lang)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if hasattr(tokenizer, "lang_code_to_id"):
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[args.target_lang]
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                module.strip()
                for module in args.lora_target_modules.split(",")
                if module.strip()
            ],
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        model_inputs = tokenizer(
            batch["source_text"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing translation pairs",
    )

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_predictions = [pred.strip() for pred in decoded_predictions]
        decoded_labels = [label.strip() for label in decoded_labels]

        bleu_result = sacrebleu.compute(
            predictions=decoded_predictions,
            references=[[label] for label in decoded_labels],
        )
        chrf_result = chrf.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
        )
        return {
            "bleu": round(bleu_result["score"], 4),
            "chrf": round(chrf_result["score"], 4),
        }

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = False
    try:
        import torch

        use_fp16 = torch.cuda.is_available() and not args.disable_fp16
    except ImportError:
        use_fp16 = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing_factor,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.generation_num_beams,
        fp16=use_fp16,
        seed=args.seed,
        report_to="none",
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    metrics = trainer.evaluate(
        max_length=args.max_target_length,
        num_beams=args.generation_num_beams,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
