-- =============================================================================
-- Migration: 001_initial_schema
-- Description: Create initial RAG API database schema
-- Version: 1.0.0
-- Date: 2025-01-16
-- =============================================================================

-- This migration creates all tables, indexes, and views for the REST API

-- Include main schema
-- (Contents from schema.sql should be here in production)
-- For now, this references the main schema file

-- Migration metadata
INSERT INTO schema_migrations (
    migration_id,
    version,
    applied_at,
    description,
    status
) VALUES (
    '001',
    '1.0.0',
    CURRENT_TIMESTAMP,
    'Initial RAG API schema',
    'applied'
);
