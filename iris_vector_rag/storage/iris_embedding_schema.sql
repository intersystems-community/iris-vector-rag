-- ============================================================================
-- IRIS EMBEDDING Schema Extensions
-- Feature: 051-add-native-iris
-- Purpose: Extend %Embedding.Config table with additional configuration fields
--          and create indexes for optimized config lookups
-- ============================================================================

-- Note: %Embedding.Config table is a system table in IRIS 2025.3+
-- This schema file documents the expected structure and adds custom indexes

-- ============================================================================
-- %Embedding.Config Table Structure (IRIS System Table)
-- ============================================================================
-- The %Embedding.Config table is provided by IRIS and has the following columns:
--
-- CREATE TABLE %Embedding.Config (
--     Name VARCHAR(255) PRIMARY KEY,           -- Unique configuration identifier
--     Configuration %Library.DynamicObject,    -- JSON configuration object
--     EmbeddingClass VARCHAR(500),            -- IRIS class for embedding generation
--     Description VARCHAR(1000)               -- Human-readable description
-- );
--
-- Configuration JSON structure (expected format):
-- {
--     "modelName": "sentence-transformers/all-MiniLM-L6-v2",
--     "hfCachePath": "/var/lib/huggingface",
--     "pythonPath": "/usr/bin/python3",
--     "batchSize": 32,
--     "devicePreference": "auto",
--     "enableEntityExtraction": false,
--     "entityTypes": ["Disease", "Symptom", "Medication"]
-- }

-- ============================================================================
-- Custom Indexes for Performance
-- ============================================================================

-- Index for fast config name lookups (if not already provided by IRIS)
-- Note: This may already exist as Name is PRIMARY KEY
-- CREATE INDEX IF NOT EXISTS idx_embedding_config_name
--     ON %Embedding.Config (Name);

-- ============================================================================
-- Helper View: Embedding Configuration Details
-- ============================================================================
-- This view expands the Configuration JSON for easier querying

CREATE OR REPLACE VIEW EmbeddingConfigDetails AS
SELECT
    Name,
    EmbeddingClass,
    Description,
    JSON_VALUE(Configuration, '$.modelName') AS ModelName,
    JSON_VALUE(Configuration, '$.hfCachePath') AS HfCachePath,
    JSON_VALUE(Configuration, '$.pythonPath') AS PythonPath,
    CAST(JSON_VALUE(Configuration, '$.batchSize') AS INTEGER) AS BatchSize,
    JSON_VALUE(Configuration, '$.devicePreference') AS DevicePreference,
    CAST(JSON_VALUE(Configuration, '$.enableEntityExtraction') AS BOOLEAN) AS EnableEntityExtraction,
    JSON_QUERY(Configuration, '$.entityTypes') AS EntityTypes
FROM %Embedding.Config;

-- ============================================================================
-- Validation Function (Optional - implemented in Python)
-- ============================================================================
-- The actual validation logic is in iris_rag/config/embedding_config.py
-- This SQL comment documents what validation checks are performed:
--
-- 1. Model file exists at hfCachePath
-- 2. Python executable exists at pythonPath
-- 3. Required Python packages installed (sentence-transformers, torch)
-- 4. If enableEntityExtraction=true, entityTypes must not be empty
-- 5. batchSize must be positive integer
-- 6. devicePreference must be one of: cuda, mps, cpu, auto

-- ============================================================================
-- Example Usage
-- ============================================================================

-- Insert a new embedding configuration
-- INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, Description)
-- VALUES (
--     'medical_embeddings_v1',
--     JSON_OBJECT(
--         'modelName', 'sentence-transformers/all-MiniLM-L6-v2',
--         'hfCachePath', '/var/lib/huggingface',
--         'pythonPath', '/usr/bin/python3',
--         'batchSize', 32,
--         'devicePreference', 'auto',
--         'enableEntityExtraction', TRUE,
--         'entityTypes', JSON_ARRAY('Disease', 'Symptom', 'Medication')
--     ),
--     '%Embedding.SentenceTransformers',
--     'Medical domain embeddings with entity extraction'
-- );

-- Query configuration details using the view
-- SELECT * FROM EmbeddingConfigDetails WHERE Name = 'medical_embeddings_v1';

-- Query configurations with entity extraction enabled
-- SELECT Name, ModelName, EntityTypes
-- FROM EmbeddingConfigDetails
-- WHERE EnableEntityExtraction = TRUE;

-- ============================================================================
-- Table Creation Example (Using EMBEDDING Column)
-- ============================================================================

-- CREATE TABLE medical_documents (
--     doc_id VARCHAR(255) PRIMARY KEY,
--     title VARCHAR(500),
--     content VARCHAR(5000),
--     source VARCHAR(255),
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--
--     -- EMBEDDING column: auto-vectorizes 'content' column
--     content_vector EMBEDDING
--         REFERENCES %Embedding.Config('medical_embeddings_v1')
--         USING content
-- );

-- When rows are inserted/updated, the EMBEDDING column automatically:
-- 1. Calls Python embedding function configured in %Embedding.Config
-- 2. Generates vector from source column (content)
-- 3. Stores vector in content_vector column
-- 4. If entity extraction enabled, extracts entities and stores in GraphRAG tables

-- ============================================================================
-- Schema Version
-- ============================================================================
-- Version: 1.0.0
-- Feature: 051-add-native-iris
-- Date: 2025-01-06
-- Compatible with: IRIS 2025.3+
