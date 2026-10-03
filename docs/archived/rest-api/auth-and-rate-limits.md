# Authentication, permissions and rate limits

## API keys

- Wire format: `Authorization: ApiKey <base64("<key_id>:<secret>")>`. `key_id` is a UUID;
  the secret is everything after the first colon.
- Storage: table `api_keys`. The secret is stored only as a **bcrypt** hash
  (`key_secret_hash`, cost 12 by default, `authentication.bcrypt_rounds`). Verification
  is `bcrypt.checkpw`.
- Fields: `key_id`, `key_secret_hash`, `name` (1 to 100), `permissions`
  (comma-separated `read,write,admin`), `rate_limit_tier`, `requests_per_minute`
  (1 to 10,000), `requests_per_hour` (1 to 100,000), `created_at`, `expires_at`
  (optional), `is_active`, `owner_email`, `description`.
- Verification order: parse header, parse base64 and UUID (401 `authentication_error`),
  load the row (`invalid_api_key` if absent), reject inactive, reject expired
  (`expired_api_key`), check the secret (`invalid_api_key`).
- Default expiry: 365 days (`authentication.default_expiration_days`).
- `AuthService` had `create_api_key`, `get_api_key`, `list_api_keys`, `revoke_api_key`, `activate_api_key` and `delete_api_key`, but no route exposed them, so
  the design never said how a key is issued (AP-21). Version 2 needs that story first.

## Permissions

`read` (query), `write` (upload and operations), `admin` (read other keys' operations).
`ApiKeyAuth.check_permission` returns 403 `authorization_error`, listing the required and
the held permissions. The WebSocket path never called it (AP-05).

## Rate-limit tiers

| Tier       | Per minute | Per hour | Concurrent |
| ---------- | ---------- | -------- | ---------- |
| basic      | 60         | 1,000    | 10         |
| premium    | 100        | 5,000    | 50         |
| enterprise | 1,000      | 50,000   | 100        |

The per-key `requests_per_minute` and `requests_per_hour` columns are what the limiter
reads; the tier table only supplies defaults when a key is created. The code enforced one
`max_concurrent_per_key` value (default 10) for everyone.

## Redis layout

| Key                                      | Type       | Meaning                                                    | TTL           |
| ---------------------------------------- | ---------- | ---------------------------------------------------------- | ------------- |
| `ratelimit:{key_id}:requests_per_minute` | sorted set | Sliding window; member and score are the request timestamp | window + 60 s |
| `ratelimit:{key_id}:requests_per_hour`   | sorted set | Same, 3,600 s window                                       | window + 60 s |
| `ratelimit:{key_id}:{quota}:exceeded`    | counter    | Times the limit was hit in the window                      | window        |
| `concurrent:{key_id}`                    | counter    | In-flight requests (INCR on entry, DECR on exit)           | 300 s         |

Algorithm per request: `ZREMRANGEBYSCORE` (drop entries older than the window), `ZCARD`
(count); if the count is at the limit, `INCR` the exceeded counter and answer 429 with
`Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining: 0` and `X-RateLimit-Reset`;
otherwise `ZADD` the current timestamp. Successful responses carry the same three
`X-RateLimit-*` headers. Concurrency: `INCR`; if above the maximum, `DECR` and reject;
release in a `finally`.

`fallback_to_memory: true` is in the config and in the start-up log, but no in-memory
limiter was written.

## Known defects of this scheme

The check and the add are separate calls, so concurrent requests overshoot; members keyed
by timestamp string can collide; a Redis outage raises and takes the API down; a crash
between `INCR` and `DECR` leaves a stale concurrent count for 300 s (AP-14). A 429 or 401
raised as `HTTPException` inside `BaseHTTPMiddleware` reaches the catch-all handler as a
500 (AP-02). Error text separated "no such key" from "wrong secret" and echoed `key_id`
(AP-10). Expiry was compared against `created_at.now()`, local time, while `created_at`
was stored in UTC (AP-12).
