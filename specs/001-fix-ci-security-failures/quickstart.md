# Quickstart: Fix CI Security Scan Failures

## Local Verification

To verify the security fixes locally, run Checkov against the repository:

```bash
# Ensure checkov is installed
pip install checkov

# Run scan on Dockerfiles
checkov -d . --framework dockerfile
```

## CI/CD Verification

1. Push the changes to the `001-fix-ci-security-failures` branch.
2. Navigate to the GitHub "Actions" tab.
3. Observe the "Security Scanning" workflow.
4. Ensure the "Infrastructure Security Scan" job passes with green status.
