-- PlantCare AI Database Schema
-- Stores curated evidence-based agricultural recommendations, 100K image dataset catalog, and diagnosis history logs

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

-- 100,000 Image Big Data Dataset Catalog Table
CREATE TABLE IF NOT EXISTS image_dataset (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_uid TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    dataset_source TEXT NOT NULL DEFAULT 'PlantVillage_Expanded_100K',
    plant_name TEXT NOT NULL,
    disease_class TEXT NOT NULL,
    split_type TEXT NOT NULL CHECK(split_type IN ('train', 'val', 'test')),
    is_augmented INTEGER NOT NULL DEFAULT 0,
    augmentation_type TEXT DEFAULT 'none',
    resolution_w INTEGER NOT NULL DEFAULT 224,
    resolution_h INTEGER NOT NULL DEFAULT 224,
    file_size_kb REAL NOT NULL,
    md5_hash TEXT,
    quality_score REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_image_class ON image_dataset(disease_class);
CREATE INDEX IF NOT EXISTS idx_image_split ON image_dataset(split_type);
CREATE INDEX IF NOT EXISTS idx_image_plant ON image_dataset(plant_name);
CREATE INDEX IF NOT EXISTS idx_image_source ON image_dataset(dataset_source);
CREATE INDEX IF NOT EXISTS idx_image_augmented ON image_dataset(is_augmented);
CREATE INDEX IF NOT EXISTS idx_image_class_split ON image_dataset(disease_class, split_type);

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

