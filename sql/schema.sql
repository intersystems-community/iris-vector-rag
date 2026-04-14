-- IVR RAG Schema — single source of truth for Python IVR and ObjectScript SDK
-- Execute each statement separated by GO delimiter

CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (
    doc_id       VARCHAR(64)               NOT NULL,
    title        VARCHAR(500),
    text_content LONGVARCHAR,
    metadata     LONGVARCHAR,
    embedding    VECTOR(DOUBLE, 384),
    PRIMARY KEY (doc_id)
)
GO

CREATE TABLE IF NOT EXISTS RAG.DocumentChunks (
    chunk_id       VARCHAR(64)               NOT NULL,
    source_doc_id  VARCHAR(64),
    chunk_text     LONGVARCHAR,
    chunk_embedding VECTOR(DOUBLE, 384),
    PRIMARY KEY (chunk_id)
)
GO

CREATE TABLE IF NOT EXISTS RAG.Entities (
    entity_id    VARCHAR(64)               NOT NULL,
    entity_name  VARCHAR(500),
    entity_type  VARCHAR(100),
    source_doc_id VARCHAR(64),
    PRIMARY KEY (entity_id)
)
GO
