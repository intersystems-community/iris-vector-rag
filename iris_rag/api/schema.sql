-- =============================================================================
-- RAG API Database Schema
-- =============================================================================
-- IRIS SQL schema for REST API tables
-- Implements entities from specs/042-full-rest-api/data-model.md
-- =============================================================================

-- =============================================================================
-- API Keys Table (Entity 3: Authentication Token)
-- =============================================================================
CREATE TABLE IF NOT EXISTS api_keys (
    key_id VARCHAR(36) PRIMARY KEY,
    key_secret_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100) NOT NULL,
    permissions VARCHAR(255) NOT NULL,  -- Comma-separated: read,write,admin
    rate_limit_tier VARCHAR(20) NOT NULL,  -- basic, premium, enterprise
    requests_per_minute INT NOT NULL,
    requests_per_hour INT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    is_active SMALLINT DEFAULT 1,  -- 1=active, 0=inactive
    owner_email VARCHAR(255) NOT NULL,
    description VARCHAR(500)
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys(owner_email);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);

-- =============================================================================
-- API Request Logs Table (Entity 1: API Request)
-- =============================================================================
CREATE TABLE IF NOT EXISTS api_request_logs (
    request_id VARCHAR(36) PRIMARY KEY,
    api_key_id VARCHAR(36),
    method VARCHAR(10) NOT NULL,  -- GET, POST, PUT, DELETE
    endpoint VARCHAR(500) NOT NULL,
    query_params TEXT,  -- JSON string
    status_code INT NOT NULL,
    execution_time_ms INT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    client_ip VARCHAR(45),
    user_agent VARCHAR(500),
    error_message VARCHAR(1000),

    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
);

-- Indexes for analytics queries
CREATE INDEX IF NOT EXISTS idx_request_logs_api_key ON api_request_logs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON api_request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON api_request_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_request_logs_status ON api_request_logs(status_code);

-- =============================================================================
-- Rate Limit Quotas Table (Entity 4: Rate Limit Quota)
-- =============================================================================
-- Note: This table is primarily for historical tracking
-- Active rate limiting uses Redis for performance
CREATE TABLE IF NOT EXISTS rate_limit_history (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    api_key_id VARCHAR(36) NOT NULL,
    quota_type VARCHAR(50) NOT NULL,  -- requests_per_minute, requests_per_hour
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    limit_value INT NOT NULL,
    requests_count INT NOT NULL,
    exceeded_count INT DEFAULT 0,

    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
);

-- Index for quota analytics
CREATE INDEX IF NOT EXISTS idx_rate_limit_api_key ON rate_limit_history(api_key_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_window ON rate_limit_history(window_start, window_end);

-- =============================================================================
-- Document Upload Operations Table (Entity 6: Document Upload Operation)
-- =============================================================================
CREATE TABLE IF NOT EXISTS document_upload_operations (
    operation_id VARCHAR(36) PRIMARY KEY,
    api_key_id VARCHAR(36) NOT NULL,
    status VARCHAR(20) NOT NULL,  -- pending, validating, processing, completed, failed
    created_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    total_documents INT NOT NULL,
    processed_documents INT DEFAULT 0,
    failed_documents INT DEFAULT 0,
    progress_percentage DECIMAL(5,2) DEFAULT 0.0,
    file_size_bytes BIGINT NOT NULL,
    pipeline_type VARCHAR(50) NOT NULL,  -- basic, basic_rerank, crag, graphrag, pylate_colbert
    validation_errors TEXT,  -- Pipe-separated list
    error_message VARCHAR(1000),

    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
);

-- Indexes for operation tracking
CREATE INDEX IF NOT EXISTS idx_upload_ops_api_key ON document_upload_operations(api_key_id);
CREATE INDEX IF NOT EXISTS idx_upload_ops_status ON document_upload_operations(status);
CREATE INDEX IF NOT EXISTS idx_upload_ops_created ON document_upload_operations(created_at);

-- =============================================================================
-- WebSocket Sessions Table (Entity 7: WebSocket Session)
-- =============================================================================
CREATE TABLE IF NOT EXISTS websocket_sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    api_key_id VARCHAR(36) NOT NULL,
    connected_at TIMESTAMP NOT NULL,
    last_activity_at TIMESTAMP NOT NULL,
    disconnected_at TIMESTAMP,
    client_ip VARCHAR(45),
    subscription_type VARCHAR(50) NOT NULL,  -- query_streaming, document_upload, all
    is_active SMALLINT DEFAULT 1,
    message_count INT DEFAULT 0,
    reconnection_token VARCHAR(100),

    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id)
);

-- Indexes for session management
CREATE INDEX IF NOT EXISTS idx_websocket_api_key ON websocket_sessions(api_key_id);
CREATE INDEX IF NOT EXISTS idx_websocket_active ON websocket_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_websocket_last_activity ON websocket_sessions(last_activity_at);

-- =============================================================================
-- Query Responses Table (Entity 5: Query Response)
-- =============================================================================
-- This table stores query execution metadata (not full responses)
-- Full responses are returned via API, this is for analytics/audit
CREATE TABLE IF NOT EXISTS query_responses (
    response_id VARCHAR(36) PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    api_key_id VARCHAR(36),
    pipeline_name VARCHAR(50) NOT NULL,
    query_text TEXT NOT NULL,
    answer_length INT,
    documents_retrieved INT,
    execution_time_ms INT NOT NULL,
    retrieval_time_ms INT,
    generation_time_ms INT,
    tokens_used INT,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP NOT NULL,

    FOREIGN KEY (api_key_id) REFERENCES api_keys(key_id),
    FOREIGN KEY (request_id) REFERENCES api_request_logs(request_id)
);

-- Indexes for analytics
CREATE INDEX IF NOT EXISTS idx_query_responses_api_key ON query_responses(api_key_id);
CREATE INDEX IF NOT EXISTS idx_query_responses_pipeline ON query_responses(pipeline_name);
CREATE INDEX IF NOT EXISTS idx_query_responses_created ON query_responses(created_at);

-- =============================================================================
-- Pipeline Health Status Table (Entity 2: Pipeline Instance)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pipeline_health_status (
    pipeline_name VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) NOT NULL,  -- healthy, degraded, unavailable
    version VARCHAR(20),
    last_health_check TIMESTAMP NOT NULL,
    total_queries BIGINT DEFAULT 0,
    avg_latency_ms DECIMAL(10,2) DEFAULT 0.0,
    error_rate DECIMAL(5,4) DEFAULT 0.0,  -- 0.0 to 1.0
    error_message VARCHAR(500),
    capabilities TEXT,  -- Comma-separated list
    description VARCHAR(500)
);

-- =============================================================================
-- Component Health Status Table (Entity 8: Health Status)
-- =============================================================================
CREATE TABLE IF NOT EXISTS component_health_status (
    component_name VARCHAR(100) PRIMARY KEY,
    status VARCHAR(20) NOT NULL,  -- healthy, degraded, unavailable
    last_checked_at TIMESTAMP NOT NULL,
    response_time_ms INT,
    version VARCHAR(50),
    dependencies TEXT,  -- Comma-separated list
    error_message VARCHAR(500),
    metrics TEXT  -- JSON string for component-specific metrics
);

-- =============================================================================
-- Database Cleanup Job Configuration
-- =============================================================================
-- This table tracks automated cleanup job status
CREATE TABLE IF NOT EXISTS cleanup_job_status (
    job_name VARCHAR(100) PRIMARY KEY,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    rows_deleted BIGINT,
    status VARCHAR(20),  -- success, failed, running
    error_message VARCHAR(500)
);

-- Initialize cleanup job entries
INSERT INTO cleanup_job_status (job_name, status)
VALUES ('api_request_logs_cleanup', 'pending')
ON DUPLICATE KEY UPDATE job_name = job_name;

INSERT INTO cleanup_job_status (job_name, status)
VALUES ('rate_limit_history_cleanup', 'pending')
ON DUPLICATE KEY UPDATE job_name = job_name;

INSERT INTO cleanup_job_status (job_name, status)
VALUES ('websocket_sessions_cleanup', 'pending')
ON DUPLICATE KEY UPDATE job_name = job_name;

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- Active API keys with request counts
CREATE VIEW IF NOT EXISTS v_api_keys_summary AS
SELECT
    k.key_id,
    k.name,
    k.owner_email,
    k.rate_limit_tier,
    k.is_active,
    k.created_at,
    k.expires_at,
    COUNT(DISTINCT r.request_id) AS total_requests,
    MAX(r.timestamp) AS last_request_at
FROM api_keys k
LEFT JOIN api_request_logs r ON k.key_id = r.api_key_id
GROUP BY k.key_id, k.name, k.owner_email, k.rate_limit_tier,
         k.is_active, k.created_at, k.expires_at;

-- Pipeline performance metrics
CREATE VIEW IF NOT EXISTS v_pipeline_metrics AS
SELECT
    pipeline_name,
    COUNT(*) AS total_queries,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    MIN(execution_time_ms) AS min_execution_time_ms,
    MAX(execution_time_ms) AS max_execution_time_ms,
    AVG(tokens_used) AS avg_tokens_used,
    AVG(documents_retrieved) AS avg_documents_retrieved
FROM query_responses
GROUP BY pipeline_name;

-- Recent errors view
CREATE VIEW IF NOT EXISTS v_recent_errors AS
SELECT
    r.request_id,
    r.timestamp,
    r.method,
    r.endpoint,
    r.status_code,
    r.error_message,
    k.name AS api_key_name,
    k.owner_email
FROM api_request_logs r
LEFT JOIN api_keys k ON r.api_key_id = k.key_id
WHERE r.status_code >= 400
ORDER BY r.timestamp DESC;

-- =============================================================================
-- Comments for Documentation
-- =============================================================================

COMMENT ON TABLE api_keys IS 'API key credentials with bcrypt-hashed secrets';
COMMENT ON TABLE api_request_logs IS 'Complete audit log of all API requests';
COMMENT ON TABLE document_upload_operations IS 'Async document upload operation tracking';
COMMENT ON TABLE websocket_sessions IS 'Active WebSocket connection tracking';
COMMENT ON TABLE query_responses IS 'Query execution metadata for analytics';
COMMENT ON TABLE pipeline_health_status IS 'Pipeline health monitoring';
COMMENT ON TABLE component_health_status IS 'System component health tracking';
COMMENT ON TABLE cleanup_job_status IS 'Automated cleanup job tracking';

-- =============================================================================
-- Schema Version
-- =============================================================================
CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL,
    description VARCHAR(500)
);

INSERT INTO schema_version (version, applied_at, description)
VALUES ('1.0.0', CURRENT_TIMESTAMP, 'Initial RAG API schema')
ON DUPLICATE KEY UPDATE version = version;
