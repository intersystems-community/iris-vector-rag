# WebSocket protocol (`/ws`)

## Handshake

1. Client connects to `/ws`; the server accepts immediately.
2. Client's **first message** must be JSON:
   `{"api_key": "<base64(key_id:secret)>", "subscription_type": "query_streaming" |
"document_upload" | "all"}` (`WebSocketAuthMessage`).
3. The server verifies the key, enforces `max_connections_per_key` (default 10), creates a
   `WebSocketSession` (`session_id`, `api_key_id`, `connected_at`, `last_activity_at`,
   `client_ip`, `subscription_type`, `is_active`, `message_count`, `reconnection_token`)
   and starts a heartbeat task. For `document_upload` and `all` subscriptions it also
   starts `watch_all_uploads` for that key.

There was no timeout on step 2 (AP-15).

## Client messages (after authentication)

| `type`         | Fields                                                                   | Needs subscription         | Effect                             |
| -------------- | ------------------------------------------------------------------------ | -------------------------- | ---------------------------------- |
| `query`        | `query`, `pipeline` (default `basic`), `top_k` (default 5), `request_id` | `query_streaming` or `all` | Runs the query and streams events  |
| `watch_upload` | `operation_id`, `request_id`                                             | `document_upload` or `all` | Streams progress for one operation |
| `pong`         | none                                                                     | any                        | Answers the server's `ping`        |

Anything else returns an `error` event. A missing subscription returns an `error` event
and the message is dropped.

## Server events

Envelope (`WebSocketEvent`): `{"event": <EventType>, "data": {...}, "timestamp":
<ISO 8601>, "request_id": <uuid>}`.

| `event`                    | `data`                                                                          |
| -------------------------- | ------------------------------------------------------------------------------- |
| `query_start`              | `query`, `pipeline`                                                             |
| `retrieval_progress`       | `documents_retrieved`                                                           |
| `generation_chunk`         | `generation_chunk` (partial answer), `is_final`                                 |
| `query_complete`           | `query`, `pipeline`, `total_documents`, `execution_time_ms`, `is_final: true`   |
| `document_upload_progress` | `operation_id`, `processed_documents`, `total_documents`, `progress_percentage` |
| `error`                    | `error` (text)                                                                  |
| `ping` / `pong`            | empty                                                                           |

## Heartbeat and lifetime

The server sends `ping` every `heartbeat_interval` (30 s) and closes a session idle longer
than `idle_timeout` (300 s). The config also lists `ping_timeout` 10 s, `max_queue_size`
1000 and reconnection with a token valid 10 minutes; the reconnection token is stored on
the session but no code redeems it.

## Defects to design out

`watch_upload` accepted any `operation_id` without checking the owner (AP-04). The
WebSocket path skipped `check_permission` and the REST input limits and echoed `str(e)`
(AP-05). Tasks from `asyncio.create_task` were not retained (AP-15). Sync pipeline calls
ran on the event loop (AP-13). `ConnectionManager` built a throwaway `ApiKeyAuth` instead
of using the one it was given (AP-20).
