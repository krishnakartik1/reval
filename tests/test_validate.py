"""Tests for dataset validation."""

import json
from pathlib import Path

import pytest

from reval.validate import load_schema, validate_dataset, validate_entry, validate_file

EVALS_DIR = Path(__file__).parent.parent / "evals"
SCHEMA_PATH = EVALS_DIR / "schema.json"


@pytest.fixture
def schema():
    return load_schema(SCHEMA_PATH)


class TestLoadSchema:
    def test_loads_valid_schema(self):
        schema = load_schema(SCHEMA_PATH)
        assert schema["title"] == "REVAL Evaluation Entry"
        assert "properties" in schema

    def test_has_required_fields(self):
        schema = load_schema(SCHEMA_PATH)
        assert "id" in schema["required"]
        assert "category" in schema["required"]


class TestValidateEntry:
    def test_valid_factual_accuracy(self, schema):
        entry = {
            "id": "us-factual_accuracy-001",
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "counterfactual_prompt": "cf test",
            "ground_truth": {
                "level": 1,
                "claim": "test",
                "correct_response": "test",
                "citations": [{"source": "CDC"}],
            },
        }
        errors = validate_entry(entry, schema, "us-factual_accuracy-001")
        assert errors == []

    def test_invalid_entry_missing_id(self, schema):
        entry = {
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
        }
        errors = validate_entry(entry, schema, "unknown")
        assert len(errors) > 0

    def test_invalid_id_pattern(self, schema):
        entry = {
            "id": "INVALID",
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "counterfactual_prompt": "cf",
            "ground_truth": {
                "level": 1,
                "claim": "c",
                "correct_response": "r",
                "citations": [{"source": "src"}],
            },
        }
        errors = validate_entry(entry, schema, "INVALID")
        assert len(errors) > 0

    def test_policy_attribution_same_entities_flagged(self, schema):
        entry = {
            "id": "us-policy_attribution-001",
            "category": "policy_attribution",
            "country": "us",
            "topic": "test",
            "counterfactual_pair": {
                "prompt_a": "a",
                "prompt_b": "b",
                "entity_a": "Same",
                "entity_b": "Same",
                "policy_topic": "test",
            },
        }
        errors = validate_entry(entry, schema, "us-policy_attribution-001")
        assert any("entity_a and entity_b should differ" in e for e in errors)

    def test_factual_accuracy_missing_citations_for_level1(self, schema):
        entry = {
            "id": "us-factual_accuracy-001",
            "category": "factual_accuracy",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "counterfactual_prompt": "cf",
            "ground_truth": {
                "level": 1,
                "claim": "c",
                "correct_response": "r",
            },
        }
        errors = validate_entry(entry, schema, "us-factual_accuracy-001")
        assert any("citations" in e for e in errors)

    def test_argumentation_parity_same_positions_flagged(self, schema):
        entry = {
            "id": "us-argumentation_parity-001",
            "category": "argumentation_parity",
            "country": "us",
            "topic": "test",
            "position_a": "Same position",
            "position_b": "Same position",
        }
        errors = validate_entry(entry, schema, "us-argumentation_parity-001")
        assert any("position_a and position_b should be different" in e for e in errors)

    def test_bad_id_format(self, schema):
        entry = {
            "id": "bad",
            "category": "issue_framing",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "rubric_id": "issue_framing",
        }
        errors = validate_entry(entry, schema, "bad")
        assert any("ID format" in e for e in errors)

    def test_empty_id_flagged(self, schema):
        entry = {
            "id": "",
            "category": "issue_framing",
            "country": "us",
            "topic": "test",
        }
        errors = validate_entry(entry, schema, "")
        assert any("Missing id" in e or "id" in e.lower() for e in errors)


class TestValidateFile:
    def test_valid_file(self, schema, tmp_path):
        f = tmp_path / "test.jsonl"
        entry = {
            "id": "us-issue_framing-001",
            "category": "issue_framing",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "rubric_id": "issue_framing",
        }
        f.write_text(json.dumps(entry) + "\n")
        valid, invalid, errors = validate_file(f, schema)
        assert valid == 1
        assert invalid == 0

    def test_invalid_file(self, schema, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"bad": True}) + "\n")
        valid, invalid, errors = validate_file(f, schema)
        assert invalid == 1
        assert len(errors) > 0

    def test_corrupt_file(self, schema, tmp_path):
        f = tmp_path / "corrupt.jsonl"
        f.write_text("not json\n")
        valid, invalid, errors = validate_file(f, schema)
        assert invalid == 1


class TestValidateDataset:
    def test_real_dataset_valid(self):
        result = validate_dataset(EVALS_DIR / "datasets", SCHEMA_PATH)
        assert result is True

    def test_empty_directory(self, tmp_path):
        result = validate_dataset(tmp_path, SCHEMA_PATH)
        assert result is True

    def test_directory_with_invalid_data(self, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text(json.dumps({"id": "BAD"}) + "\n")
        result = validate_dataset(tmp_path, SCHEMA_PATH)
        assert result is False

    def test_verbose_output(self, tmp_path):
        f = tmp_path / "test.jsonl"
        entry = {
            "id": "us-issue_framing-001",
            "category": "issue_framing",
            "country": "us",
            "topic": "test",
            "prompt": "test",
            "rubric_id": "issue_framing",
        }
        f.write_text(json.dumps(entry) + "\n")
        result = validate_dataset(tmp_path, SCHEMA_PATH, verbose=True)
        assert result is True
