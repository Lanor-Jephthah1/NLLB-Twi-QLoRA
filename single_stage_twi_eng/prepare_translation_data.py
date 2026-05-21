from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from translation_config import DATASET_CONFIGS, get_dataset_config

try:
    from ftfy import fix_text as ftfy_fix_text
except ImportError:  # pragma: no cover - optional dependency
    ftfy_fix_text = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and split the Ghana NLP translation CSV datasets for model fine-tuning."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("ghana_nlp_translation_datasets"),
        help="Folder containing the downloaded Ghana NLP dataset subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training/processed_data"),
        help="Where cleaned train/validation JSONL files should be written.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_CONFIGS)],
        default="all",
        help="Which dataset to process.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before train/validation split.",
    )
    return parser.parse_args()


def fix_mojibake(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    if ftfy_fix_text is not None:
        cleaned = ftfy_fix_text(cleaned)

    cleaned = cleaned.replace("\ufeff", "")
    cleaned = " ".join(cleaned.split())
    return cleaned


def row_is_usable(source_text: str, target_text: str) -> bool:
    if not source_text or not target_text:
        return False
    if source_text == target_text:
        return False
    return True


def load_rows(csv_path: Path, source_column: str, target_column: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            source_text = fix_mojibake(raw.get(source_column, ""))
            target_text = fix_mojibake(raw.get(target_column, ""))
            if not row_is_usable(source_text, target_text):
                continue
            rows.append(
                {
                    "id": str(raw.get("id", "")).strip(),
                    "source_text": source_text,
                    "target_text": target_text,
                }
            )
    return rows


def train_validation_split(
    rows: list[dict[str, str]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)

    validation_count = max(1, int(len(shuffled) * validation_ratio))
    validation_rows = shuffled[:validation_count]
    train_rows = shuffled[validation_count:]
    return train_rows, validation_rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_dataset(
    dataset_name: str,
    datasets_root: Path,
    output_root: Path,
    validation_ratio: float,
    seed: int,
) -> dict[str, object]:
    config = get_dataset_config(dataset_name)
    csv_path = datasets_root / config.dataset_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    rows = load_rows(csv_path, config.source_column, config.target_column)
    if len(rows) < 2:
        raise ValueError(f"Not enough usable rows found in {csv_path}")

    train_rows, validation_rows = train_validation_split(rows, validation_ratio, seed)
    for row in train_rows:
        row.update(
            {
                "source_lang": config.source_lang,
                "target_lang": config.target_lang,
                "dataset_name": dataset_name,
            }
        )
    for row in validation_rows:
        row.update(
            {
                "source_lang": config.source_lang,
                "target_lang": config.target_lang,
                "dataset_name": dataset_name,
            }
        )

    dataset_output_dir = output_root / dataset_name.lower()
    write_jsonl(dataset_output_dir / "train.jsonl", train_rows)
    write_jsonl(dataset_output_dir / "validation.jsonl", validation_rows)

    summary = {
        "dataset_name": dataset_name,
        "dataset_dir": config.dataset_dir,
        "source_label": config.source_label,
        "target_label": config.target_label,
        "source_lang": config.source_lang,
        "target_lang": config.target_lang,
        "nllb_compatible": config.nllb_compatible,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "output_dir": str(dataset_output_dir),
    }
    with (dataset_output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset_name in dataset_names:
        summaries.append(
            process_dataset(
                dataset_name=dataset_name,
                datasets_root=args.datasets_root,
                output_root=output_root,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            )
        )

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)

    print(f"Wrote processed datasets to {output_root}")
    for summary in summaries:
        compatibility = "NLLB-ready" if summary["nllb_compatible"] else "needs a different base model"
        print(
            f"- {summary['dataset_name']}: "
            f"{summary['train_rows']} train / {summary['validation_rows']} validation ({compatibility})"
        )


if __name__ == "__main__":
    main()
