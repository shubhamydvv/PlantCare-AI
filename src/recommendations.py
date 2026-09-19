"""Recommendation retrieval module and SQLite diagnosis logging."""

from __future__ import annotations
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import DB_PATH, format_class_name
from database.seed_data import init_and_seed_database

logger = logging.getLogger("PlantCare.Recommendations")


def get_db_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    db_file = Path(db_path)
    if not db_file.exists():
        logger.info(f"Database not found at {db_file}. Initializing and seeding...")
        init_and_seed_database(db_path=db_file)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    return conn


def get_recommendation_for_disease(
    disease_class: str,
    db_path: Path | str = DB_PATH,
) -> Dict[str, Any]:
    """
    Retrieves evidence-based symptoms, prevention measures, and treatment guidance.
    Provides safe and documented fallback if no exact disease match exists.
    """
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT disease_class, plant_name, disease_name, symptoms, prevention, treatment, source_citation, needs_review, last_reviewed
            FROM recommendations
            WHERE disease_class = ?
            """,
            (disease_class,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "disease_class": row["disease_class"],
                "plant_name": row["plant_name"],
                "disease_name": row["disease_name"],
                "symptoms": row["symptoms"],
                "prevention": row["prevention"],
                "treatment": row["treatment"],
                "source_citation": row["source_citation"],
                "needs_review": bool(row["needs_review"]),
                "last_reviewed": row["last_reviewed"],
                "is_fallback": False,
            }
    except Exception as e:
        logger.error(f"Error querying recommendation database: {e}")

    # Documented Fallback: Format class safely and alert that review is needed
    plant, disease = format_class_name(disease_class)
    return {
        "disease_class": disease_class,
        "plant_name": plant,
        "disease_name": disease,
        "symptoms": f"Visible pathological symptoms affecting {plant} foliage.",
        "prevention": "Prune symptomatic leaves with sanitized tools; avoid overhead watering; ensure adequate air circulation and plant spacing.",
        "treatment": "Isolate affected foliage. Consult a local agricultural extension officer for current crop-specific product and dosage guidance.",
        "source_citation": "FAO / ICAR General Crop Health Guidance (Unverified Class Fallback)",
        "needs_review": True,
        "last_reviewed": datetime.now().strftime("%Y-%m-%d"),
        "is_fallback": True,
    }


def log_diagnosis_to_history(
    image_hash: str,
    plant_name: str,
    predicted_disease: str,
    confidence: float,
    model_name: str,
    flagged_uncertain: bool = False,
    notes: str = "",
    db_path: Path | str = DB_PATH,
) -> int:
    """Logs a completed diagnosis prediction to the SQLite database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO diagnosis_history (
            timestamp, image_hash, plant_name, predicted_disease, confidence, model_name, flagged_uncertain, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            image_hash,
            plant_name,
            predicted_disease,
            float(confidence),
            model_name,
            1 if flagged_uncertain else 0,
            notes,
        ),
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id or 0


def get_recent_diagnoses(
    limit: int = 10,
    db_path: Path | str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Retrieves most recent diagnoses history."""
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, image_hash, plant_name, predicted_disease, confidence, model_name, flagged_uncertain, notes
            FROM diagnosis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to fetch diagnosis history: {e}")
        return []
