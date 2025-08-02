# Daemon Performance Optimization

## Overview

This document describes the critical performance optimization implemented for the daemon controller to eliminate 5-minute test delays caused by hardcoded retry intervals.

## Problem Statement

The original [`daemon_controller.py`](../iris_rag/controllers/reconciliation_components/daemon_controller.py) implementation had hardcoded 5-minute (300-second) error retry intervals that caused massive delays in test environments:

```python
# Original problematic code
self.error_retry_interval_seconds = reconciliation_config.get('error_retry_minutes', 5) * 60  # 300 seconds!
```

When tests failed (common in test scenarios), the daemon would wait 5 full minutes before the next iteration, causing:
- Test suites taking 5+ minutes instead of seconds
- Blocked development productivity
- Frustrated developers waiting for test feedback

## Solution: Environment-Aware Configuration

### 1. Environment Detection Utility

Created [`common/environment_utils.py`](../common/environment_utils.py) with intelligent environment detection:

```python
def detect_environment() -> EnvironmentType:
    """
    Detect the current execution environment.
    
    Detection logic:
    1. If pytest is running -> "test"
    2. If APP_ENV environment variable is set -> use that value
    3. If CI environment variables are set -> "test" 
    4. If DEBUG_MODE is true -> "development"
    5. Default -> "production"
    """
```

### 2. Environment-Specific Defaults

The optimization provides different retry intervals based on environment:

| Environment | Error Retry Interval | Default Interval | Use Case |
|-------------|---------------------|------------------|----------|
| **Test** | 1 second | 1 second | Fast test execution |
| **Development** | 30 seconds | 5 minutes | Reasonable dev feedback |
| **Production** | 5 minutes | 1 hour | Robust production operation |

### 3. Updated Daemon Controller

The daemon controller now uses environment-aware defaults:

```python
# New optimized code
from common.environment_utils import get_daemon_retry_interval, get_daemon_default_interval, detect_environment

# In __init__:
current_env = detect_environment()
self.error_retry_interval_seconds = get_daemon_retry_interval(
    config_error_retry_minutes * 60 if current_env == "production" else None
)
```

## Configuration Options

### Environment Variables

You can override defaults using environment variables:

```bash
# Override error retry interval (seconds)
export DAEMON_ERROR_RETRY_SECONDS=1

# Override default interval (seconds)  
export DAEMON_DEFAULT_INTERVAL_SECONDS=3600

# Set explicit environment
export APP_ENV=test
```

### Configuration File

Traditional configuration still works for production:

```yaml
reconciliation:
  interval_hours: 1
  error_retry_minutes: 5
```

## Performance Impact

### Before Optimization
- Test with error: **5+ minutes** (300-second retry)
- Test suite: **Multiple 5-minute delays**
- Developer productivity: **Severely impacted**

### After Optimization
- Test with error: **~1 second** (1-second retry)
- Test suite: **10-15 seconds total**
- Developer productivity: **Restored**

### Test Results
```
# Before: Tests would hang for 5+ minutes
Using shorter retry interval due to error: 300 seconds

# After: Tests complete quickly
DaemonController initialized for test environment
Default interval: 1s, Error retry: 1s
Using shorter retry interval due to error: 1 seconds
```

## Backward Compatibility

The optimization maintains full backward compatibility:

1. **Production environments** retain original 5-minute retry intervals
2. **Existing configuration** continues to work unchanged
3. **Manual overrides** still function as expected
4. **API compatibility** is preserved

## Usage Examples

### Test Environment (Automatic)
```python
# When running pytest, automatically uses 1-second intervals
python -m pytest tests/test_reconciliation_daemon.py
```

### Development Environment
```bash
export APP_ENV=development
# Uses 30-second error retry, 5-minute default interval
```

### Production Environment
```bash
export APP_ENV=production
# Uses 5-minute error retry, 1-hour default interval
```

### Manual Override
```bash
export DAEMON_ERROR_RETRY_SECONDS=10
# Forces 10-second retry regardless of environment
```

## Implementation Details

### Environment Detection Logic

1. **Pytest Detection**: Checks for `pytest` in `sys.modules` or `PYTEST_CURRENT_TEST` environment variable
2. **CI Detection**: Looks for common CI environment variables (`CI`, `GITLAB_CI`, `GITHUB_ACTIONS`, etc.)
3. **Explicit Setting**: Honors `APP_ENV` environment variable
4. **Debug Mode**: Uses `DEBUG_MODE` environment variable
5. **Safe Default**: Defaults to "production" for safety

### Configuration Hierarchy

1. **Explicit parameter override** (highest priority)
2. **Environment variable override**
3. **Environment-specific default**
4. **Configuration file setting**
5. **Hardcoded fallback** (lowest priority)

## Testing

The optimization includes comprehensive tests:

```bash
# Test the optimization
python -m pytest tests/test_reconciliation_daemon.py::TestReconciliationDaemon::test_daemon_error_handling_and_retry_interval -v

# Test full daemon suite
python -m pytest tests/test_reconciliation_daemon.py -v
```

## Monitoring

The daemon controller logs environment detection for visibility:

```
INFO - DaemonController initialized for test environment
INFO - Default interval: 1s, Error retry: 1s
```

## Security Considerations

- Environment detection is safe and doesn't expose sensitive information
- Production defaults remain conservative (5-minute retries)
- No security-sensitive configuration is auto-detected

## Future Enhancements

Potential future improvements:

1. **Adaptive retry intervals** based on error types
2. **Exponential backoff** for repeated failures
3. **Circuit breaker patterns** for persistent issues
4. **Metrics collection** for retry interval effectiveness

## Conclusion

This optimization eliminates a critical development productivity blocker while maintaining production robustness. Tests now complete in seconds instead of minutes, dramatically improving the developer experience without compromising production reliability.