# IRIS Environment Variables

Use these environment variables to configure IRIS connections for local development and tests.

## Required/Supported Variables

- `IRIS_HOST` (default: `localhost`)
- `IRIS_PORT` (default: `1974`)
- `IRIS_NAMESPACE` (default: `USER`)
- `IRIS_USERNAME` (default: `SuperUser`)
- `IRIS_PASSWORD` (default: `SYS`)

## Test Execution Notes

- Tests are expected to run against **live IRIS** via **iris-devtester**.
- Do **not** hardcode ports; always resolve via `IRISContainer.attach("los-iris").get_exposed_port(1972)` when needed.
- `SKIP_IRIS_TESTS` defaults to `false` and should not be set to `true` in normal runs.
