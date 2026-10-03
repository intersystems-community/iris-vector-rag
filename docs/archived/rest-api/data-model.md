# Data model

Source: `original/schema.sql` (286 lines, IRIS SQL, every statement `IF NOT EXISTS`).
Identifiers are `VARCHAR(36)` UUID strings.

## Tables

| Table                        | Purpose                                    | Notable columns                                                                                                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api_keys`                   | Credentials                                | `key_id` PK, `key_secret_hash`, `permissions`, `rate_limit_tier`, `requests_per_minute`, `requests_per_hour`, `expires_at`, `is_active`, `owner_email`                                                                                                               |
| `api_request_logs`           | One row per request                        | `request_id` PK, `api_key_id` FK, `method`, `endpoint`, `query_params`, `status_code`, `execution_time_ms`, `timestamp`, `client_ip`, `user_agent`, `error_message`                                                                                                  |
| `rate_limit_history`         | Window summaries (Redis is the live store) | `api_key_id` FK, `quota_type`, `window_start`, `window_end`, `limit_value`, `requests_count`, `exceeded_count`                                                                                                                                                       |
| `document_upload_operations` | Upload operation and progress              | `operation_id` PK, `api_key_id` FK, `status`, `created_at`, `started_at`, `completed_at`, `total_documents`, `processed_documents`, `failed_documents`, `progress_percentage DECIMAL(5,2)`, `file_size_bytes`, `pipeline_type`, `validation_errors`, `error_message` |
| `websocket_sessions`         | Session audit                              | `session_id` PK, `api_key_id` FK, `connected_at`, `last_activity_at`, `disconnected_at`, `subscription_type`, `is_active`, `message_count`, `reconnection_token`                                                                                                     |
| `query_responses`            | Response summary per query                 | `response_id` PK, `request_id`, `pipeline_name`, `query_text`, `answer_length`, `documents_retrieved`, timings, `tokens_used`                                                                                                                                        |
| `pipeline_health_status`     | Latest health per pipeline                 | see `original/schema.sql`                                                                                                                                                                                                                                            |
| `component_health_status`    | Latest health per component                | see `original/schema.sql`                                                                                                                                                                                                                                            |
| `cleanup_job_status`         | Retention job bookkeeping                  | see `original/schema.sql`                                                                                                                                                                                                                                            |
| `schema_version`             | Applied schema version                     | `version`, `applied_at`, `description`                                                                                                                                                                                                                               |

Views: `v_api_keys_summary`, `v_pipeline_metrics`, `v_recent_errors`.

Each table has indexes on its foreign key and its time column (for example
`idx_request_logs_timestamp`, `idx_upload_ops_status`).

## Operation and upload-progress model

State machine: `pending`, then `validating`, then `processing`, then `completed` or
`failed`. `progress_percentage = processed_documents / total_documents * 100`.
`validation_errors` is stored as a pipe-separated string. The service saved the row with
`_save_operation` (a synchronous IRIS write inside an async handler, AP-13) and updated it
after each document. Retention: `operation_retention_days` 7; request logs 30 days, with a
midnight cleanup schedule (`0 0 * * *`).

## Migration

`original/001_initial_schema.sql` is a stub, not a migration: its body is comments that
say "contents of `schema.sql` should be here", followed by an `INSERT` into a
`schema_migrations` table that `schema.sql` never creates (`schema.sql` creates
`schema_version`). No runner applied either file. Version 2 needs one migration mechanism
and the choice to keep operational state in IRIS tables or elsewhere.

## SQL notes for a rebuild

`TEXT` columns (`query_params`, `query_text`, `validation_errors`) are CLOBs in IRIS and
cannot be used with `LIKE` or scalar functions; use `VARCHAR(N)` for anything filtered.
Use `SELECT TOP n`, not `FETCH FIRST`. The upload `limit` was passed to SQL `LIMIT ?`
unbounded (AP-11).
