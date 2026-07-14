from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "qa_dataset.json"


def test_eval_dataset_is_annotated() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    assert dataset["items"]
    for item in dataset["items"]:
        assert item["id"]
        assert item["question"]
        assert item["reference_answer"]
        assert item["expected_event_uids"]
        assert item["expected_keywords"]
