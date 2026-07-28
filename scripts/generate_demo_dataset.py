from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

SPLITS = {
    "pretrain": DATA_ROOT / "demo_assets" / "pretrain",
    "finetune": DATA_ROOT / "demo_assets" / "finetune",
    "test": DATA_ROOT / "demo_assets" / "test",
}

PARTITIONS = {
    "pretrain": [
        DATA_ROOT / "partitions" / "pretrain" / "train.jsonl",
        DATA_ROOT / "partitions" / "pretrain" / "valid.jsonl",
        DATA_ROOT / "partitions" / "pretrain" / "test.jsonl",
    ],
    "finetune": [
        DATA_ROOT / "partitions" / "finetune" / "train_filtered.jsonl",
        DATA_ROOT / "partitions" / "finetune" / "test_filtered.jsonl",
    ],
    "test": [
        DATA_ROOT / "partitions" / "test" / "test_filtered.jsonl",
    ],
}

ABC_TEMPLATE = """X:1
T:Demo {index}
M:4/4
L:1/8
Q:1/4=120
K:C
V:1
C D E F | G A B c | d c B A | G F E D |
V:2
E F G A | B c d e | f e d c | B A G F |
"""


def build_abc_text(index: int) -> str:
    return ABC_TEMPLATE.format(index=index)


def build_spectrogram(index: int) -> np.ndarray:
    rng = np.random.default_rng(20260728 + index)
    # Keep the demo samples small but shape-compatible with the model.
    return rng.normal(size=(1025, 96)).astype(np.float32)


def write_assets(split_name: str, asset_dir: Path) -> list[dict[str, str]]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for index in range(1, 6):
        spec_path = asset_dir / f"{split_name}_demo_{index}.npy"
        abc_path = asset_dir / f"{split_name}_demo_{index}.abc"

        np.save(spec_path, build_spectrogram(index))
        abc_path.write_text(build_abc_text(index), encoding="utf-8")

        records.append(
            {
                "spectrogram": str(spec_path.relative_to(PROJECT_ROOT)),
                "output": str(abc_path.relative_to(PROJECT_ROOT)),
            }
        )

    return records


def write_partitions(records_by_split: dict[str, list[dict[str, str]]]) -> None:
    for split_name, files in PARTITIONS.items():
        records = records_by_split[split_name]
        for jsonl_path in files:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    records_by_split = {}
    for split_name, asset_dir in SPLITS.items():
        records_by_split[split_name] = write_assets(split_name, asset_dir)

    write_partitions(records_by_split)
    print("Demo dataset generated successfully.")


if __name__ == "__main__":
    main()
