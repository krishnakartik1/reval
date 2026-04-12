"""Tests for JSONL dataset loading and JSON schema validation."""

import json
from pathlib import Path

import jsonschema
import pytest

from reval.contracts import EvalCategory
from reval.runner import load_evals_from_directory, load_evals_from_jsonl

EVALS_DIR = Path(__file__).parent.parent / "evals"
DATASETS_DIR = EVALS_DIR / "datasets"
SCHEMA_PATH = EVALS_DIR / "schema.json"


# ---------------------------------------------------------------------------
# Load all JSONL datasets
# ---------------------------------------------------------------------------


class TestLoadDatasets:
    def test_load_us_factual_accuracy(self):
        entries = load_evals_from_jsonl(DATASETS_DIR / "us" / "factual_accuracy.jsonl")
        assert len(entries) == 5
        for e in entries:
            assert e.category == EvalCategory.FACTUAL_ACCURACY
            assert e.counterfactual_prompt is not None
            assert e.ground_truth is not None
            assert e.prompt is not None

    def test_load_india_factual_accuracy(self):
        entries = load_evals_from_jsonl(
            DATASETS_DIR / "india" / "factual_accuracy.jsonl"
        )
        assert len(entries) == 5
        for e in entries:
            assert e.counterfactual_prompt is not None

    def test_load_us_figure_treatment(self):
        entries = load_evals_from_jsonl(DATASETS_DIR / "us" / "figure_treatment.jsonl")
        assert len(entries) == 4
        for e in entries:
            assert e.category == EvalCategory.FIGURE_TREATMENT
            assert e.figure_pair is not None
            assert e.figure_pair.figure_a is not None
            assert e.figure_pair.affiliation_a is not None

    def test_load_india_figure_treatment(self):
        entries = load_evals_from_jsonl(
            DATASETS_DIR / "india" / "figure_treatment.jsonl"
        )
        assert len(entries) == 3
        for e in entries:
            assert e.figure_pair is not None

    def test_load_all_datasets(self):
        entries = load_evals_from_directory(DATASETS_DIR)
        assert len(entries) > 0
        categories = {e.category for e in entries}
        assert EvalCategory.FACTUAL_ACCURACY in categories
        assert EvalCategory.FIGURE_TREATMENT in categories

    def test_filter_by_country(self):
        us_entries = load_evals_from_directory(DATASETS_DIR, country="us")
        india_entries = load_evals_from_directory(DATASETS_DIR, country="india")
        assert all(e.country.value == "us" for e in us_entries)
        assert all(e.country.value == "india" for e in india_entries)

    def test_filter_by_category(self):
        fa_entries = load_evals_from_directory(
            DATASETS_DIR, category=EvalCategory.FACTUAL_ACCURACY
        )
        assert all(e.category == EvalCategory.FACTUAL_ACCURACY for e in fa_entries)
        assert len(fa_entries) == 10  # 5 US + 5 India


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    @pytest.fixture
    def schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_factual_accuracy_requires_counterfactual_prompt(self, schema):
        entry = {
            "id": "us-factual_accuracy-001",
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
            "prompt": "test prompt",
            "ground_truth": {
                "level": 1,
                "claim": "test",
                "correct_response": "test",
            },
            # missing counterfactual_prompt
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(entry, schema)

    def test_factual_accuracy_valid(self, schema):
        entry = {
            "id": "us-factual_accuracy-001",
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
            "prompt": "test prompt",
            "counterfactual_prompt": "rephrased test prompt",
            "ground_truth": {
                "level": 1,
                "claim": "test",
                "correct_response": "test",
            },
        }
        jsonschema.validate(entry, schema)  # should not raise

    def test_figure_treatment_requires_figure_pair(self, schema):
        entry = {
            "id": "us-figure_treatment-001",
            "category": "figure_treatment",
            "country": "us",
            "topic": "test",
            "rubric_id": "figure_treatment",
            # missing figure_pair
        }
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(entry, schema)

    def test_figure_treatment_valid(self, schema):
        entry = {
            "id": "us-figure_treatment-001",
            "category": "figure_treatment",
            "country": "us",
            "topic": "test",
            "rubric_id": "figure_treatment",
            "figure_pair": {
                "prompt_a": "Describe A",
                "prompt_b": "Describe B",
                "figure_a": "Figure A",
                "figure_b": "Figure B",
                "affiliation_a": "Party X",
                "affiliation_b": "Party Y",
            },
        }
        jsonschema.validate(entry, schema)  # should not raise

    def test_figure_treatment_no_longer_requires_prompt(self, schema):
        """figure_treatment now requires figure_pair, not prompt."""
        entry = {
            "id": "us-figure_treatment-001",
            "category": "figure_treatment",
            "country": "us",
            "topic": "test",
            "rubric_id": "figure_treatment",
            "figure_pair": {
                "prompt_a": "A",
                "prompt_b": "B",
                "figure_a": "FA",
                "figure_b": "FB",
                "affiliation_a": "PX",
                "affiliation_b": "PY",
            },
        }
        jsonschema.validate(entry, schema)  # should not raise — no prompt needed

    def test_all_us_datasets_pass_schema(self, schema):
        for jsonl_file in (DATASETS_DIR / "us").glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    entry = json.loads(line)
                    jsonschema.validate(entry, schema)

    def test_all_india_datasets_pass_schema(self, schema):
        for jsonl_file in (DATASETS_DIR / "india").glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line_num, line in enumerate(f, 1):
                    entry = json.loads(line)
                    jsonschema.validate(entry, schema)
