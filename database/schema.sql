-- PlantCare AI Database Schema
-- Stores curated evidence-based agricultural recommendations and diagnosis history logs

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_class TEXT UNIQUE NOT NULL,
    plant_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    symptoms TEXT NOT NULL,
    prevention TEXT NOT NULL,
    treatment TEXT NOT NULL,
    source_citation TEXT NOT NULL,
    needs_review INTEGER NOT NULL DEFAULT 0,
    last_reviewed TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recommendations_class ON recommendations(disease_class);
CREATE INDEX IF NOT EXISTS idx_recommendations_plant ON recommendations(plant_name);

CREATE TABLE IF NOT EXISTS diagnosis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    plant_name TEXT NOT NULL,
    predicted_disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_name TEXT NOT NULL,
    flagged_uncertain INTEGER NOT NULL DEFAULT 0,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_history_timestamp ON diagnosis_history(timestamp);
