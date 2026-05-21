from __future__ import annotations

from dataclasses import dataclass


NLLB_RECOMMENDED_MODEL = "facebook/nllb-200-distilled-600M"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_dir: str
    source_lang: str
    target_lang: str
    source_label: str
    target_label: str
    source_column: str = "text"
    target_column: str = "label"
    nllb_compatible: bool = False


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "ENGLISH_TWI_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="ENGLISH_TWI_PARALLEL_TEXT",
        source_lang="eng_Latn",
        target_lang="twi_Latn",
        source_label="English",
        target_label="Twi",
        nllb_compatible=True,
    ),
    "TWI_ENGLISH_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="TWI_ENGLISH_PARALLEL_TEXT",
        source_lang="twi_Latn",
        target_lang="eng_Latn",
        source_label="Twi",
        target_label="English",
        nllb_compatible=True,
    ),
    "EWE_ENGLISH_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="EWE_ENGLISH_PARALLEL_TEXT",
        source_lang="ewe_Latn",
        target_lang="eng_Latn",
        source_label="Ewe",
        target_label="English",
        source_column="label",
        target_column="text",
        nllb_compatible=True,
    ),
    "ENGLISH_EWE_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="EWE_ENGLISH_PARALLEL_TEXT",
        source_lang="eng_Latn",
        target_lang="ewe_Latn",
        source_label="English",
        target_label="Ewe",
        source_column="text",
        target_column="label",
        nllb_compatible=True,
    ),
    "GA_ENGLISH_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="GA_ENGLISH_PARALLEL_TEXT",
        source_lang="gaa_Latn",
        target_lang="eng_Latn",
        source_label="Ga",
        target_label="English",
        nllb_compatible=False,
    ),
    "FANTE_ENGLISH_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="FANTE_ENGLISH_PARALLEL_TEXT",
        source_lang="fat_Latn",
        target_lang="eng_Latn",
        source_label="Fante",
        target_label="English",
        nllb_compatible=False,
    ),
    "KUSAAL_ENGLISH_PARALLEL_TEXT": DatasetConfig(
        dataset_dir="KUSAAL_ENGLISH_PARALLEL_TEXT",
        source_lang="kus_Latn",
        target_lang="eng_Latn",
        source_label="Kusaal",
        target_label="English",
        nllb_compatible=False,
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[dataset_name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_CONFIGS))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}") from exc
