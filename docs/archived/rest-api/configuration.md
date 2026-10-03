# Configuration

The file was `config/api_config.yaml` (157 lines; verbatim copy in
[original/api_config.yaml](original/api_config.yaml)). `main.load_config()` read
`config/api_config.yaml` relative to the working directory, and fell back to
`get_default_config()` when it was missing.

**The YAML and the code disagree.** The YAML nests pool keys under
`database.connection_pool`, puts CORS under `server.cors` and Redis under
`rate_limiting.redis_url`; the code reads `database.*`, a top-level `cors`, a top-level
`redis` and `auth.bcrypt_rounds`. Most YAML keys were therefore never read. The table
marks what the code read.

| Key                                                                                                                 | Value in YAML                                                        | Read by code                                                    |
| ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------- |
| `server.host`, `port`, `workers`, `reload`                                                                          | `0.0.0.0`, 8000, 4, false                                            | yes (`uvicorn.run`)                                             |
| `server.cors.*`                                                                                                     | enabled; localhost:3000 and :8080                                    | no (code reads top-level `cors`)                                |
| `database.connection_pool.*`                                                                                        | size 20, overflow 10, recycle 3600 s, pre-ping, timeout 30 s         | no (code reads `database.pool_size` etc.)                       |
| `database.host/port/namespace/username/password`                                                                    | absent                                                               | yes; defaults `localhost`, 1972, `USER`, `demo`, `demo` (AP-09) |
| `rate_limiting.enabled`, `redis_url`, `fallback_to_memory`                                                          | true, `${REDIS_URL:-redis://localhost:6379/0}`, true                 | no (code reads top-level `redis.*`)                             |
| `rate_limiting.tiers.{basic,premium,enterprise}`                                                                    | see [auth-and-rate-limits.md](auth-and-rate-limits.md)               | no                                                              |
| `rate_limiting.max_concurrent_per_key`                                                                              | absent                                                               | yes, default 10                                                 |
| `authentication.bcrypt_rounds`, `default_expiration_days`, `session_timeout_minutes`, `heartbeat_interval_seconds`  | 12, 365, 5, 30                                                       | no (code reads `auth.bcrypt_rounds`)                            |
| `logging.level`, `format`, `log_requests`, `log_response_bodies`, `retention_days`, `cleanup_schedule`              | INFO, json, true, false, 30, midnight                                | no                                                              |
| `pipelines.enabled`                                                                                                 | all five pipelines                                                   | yes                                                             |
| `pipelines.health_check_interval_seconds`, `health_check_timeout_seconds`                                           | 30, 5                                                                | no                                                              |
| `pipelines.timeouts.*`                                                                                              | basic 60, basic_rerank 90, crag 120, graphrag 180, pylate_colbert 90 | no                                                              |
| `websocket.max_connections_per_key`                                                                                 | 10                                                                   | yes                                                             |
| `websocket.ping_interval`, `ping_timeout`, `max_queue_size`, `allow_reconnection`, `reconnection_token_ttl_minutes` | 30, 10, 1000, true, 10                                               | no (code reads `heartbeat_interval`, `idle_timeout`)            |
| `document_upload.max_file_size_bytes`                                                                               | 104857600 (100 MB)                                                   | hard-coded in the route                                         |
| `document_upload.allowed_content_types`, `max_concurrent_uploads`, `operation_retention_days`                       | 4 types, 5, 7                                                        | partly                                                          |
| `performance.request_timeout_seconds`, `target_p95_latency_ms`, `collect_metrics`, `metrics_interval_seconds`       | 60, 2000, true, 60                                                   | no                                                              |
| `development.debug_mode`, `docs_enabled`, `docs_url`, `redoc_url`, `openapi_url`                                    | false, true, `/docs`, `/redoc`, `/openapi.json`                      | no (paths hard-coded)                                           |

Environment: only `${REDIS_URL:-...}` appears in the YAML, and the loader does no
substitution, so it was never expanded.

For a rebuild: one typed settings object (Pydantic settings), environment first, no
credential defaults, and a start-up error when a required value is missing.
