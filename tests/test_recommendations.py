"""Unit tests for SQLite recommendation database and history logger."""

import pytest
import sqlite3
from pathlib import Path

from src.config import PLANT_VILLAGE_CLASSES
from src.recommendations import (
    get_recommendation_for_disease,
    log_diagnosis_to_history,
    get_recent_diagnoses,
)
from database.seed_data import init_and_seed_database


@pytest.fixture
def temp_db(tmp_path):
    """Initializes and seeds an isolated SQLite database in a temp directory."""
    db_file = tmp_path / "test_plantcare.db"
    schema_file = Path("database/schema.sql")
    init_and_seed_database(db_path=db_file, schema_path=schema_file)
    return db_file


def test_seed_database_contains_all_classes(temp_db):
    """Verifies that all 38 PlantVillage classes have a non-null, source-cited recommendation entry."""
    for cls in PLANT_VILLAGE_CLASSES:
        rec = get_recommendation_for_disease(cls, db_path=temp_db)
        assert rec["disease_class"] == cls
        assert len(rec["symptoms"]) > 0
        assert len(rec["prevention"]) > 0
        assert len(rec["treatment"]) > 0
        assert len(rec["source_citation"]) > 0
        assert rec["is_fallback"] is False


def test_recommendation_fallback_for_unknown_disease(temp_db):
    """Verifies that querying an unknown class returns the safe, documented fallback without crashing."""
    rec = get_recommendation_for_disease("Banana___Unknown_Blight", db_path=temp_db)
    assert rec["is_fallback"] is True
    assert rec["needs_review"] is True
    assert "extension officer" in rec["treatment"].lower()


def test_diagnosis_history_logging(temp_db):
    """Verifies that diagnoses can be logged to history and retrieved."""
    row_id = log_diagnosis_to_history(
        image_hash="abc123hash",
        plant_name="Tomato",
        predicted_disease="Early Blight",
        confidence=0.945,
        model_name="resnet50",
        flagged_uncertain=False,
        notes="Automated test entry",
        db_path=temp_db,
    )
    assert row_id > 0

    recent = get_recent_diagnoses(limit=5, db_path=temp_db)
    assert len(recent) == 1
    assert recent[0]["image_hash"] == "abc123hash"
    assert recent[0]["confidence"] == 0.945
