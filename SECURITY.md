# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to: **thomas.dyar@intersystems.com**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 5 business days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 1 week
  - Medium: 2 weeks
  - Low: Next release cycle

### Scope

This security policy applies to:
- The `iris-vector-rag` Python package
- Associated Docker images
- API endpoints when deployed
- MCP server implementations

### Out of Scope

- Vulnerabilities in dependencies (report to upstream maintainers)
- Issues in user-deployed configurations
- Social engineering attacks

## Security Best Practices

When using iris-vector-rag:

1. **API Keys**: Never commit API keys or credentials to version control
2. **Database Credentials**: Use environment variables for IRIS connection strings
3. **Docker**: Run containers as non-root users in production
4. **Network**: Use TLS for all production API communications
5. **Updates**: Keep dependencies updated with `uv sync --upgrade`

## Security Features

iris-vector-rag includes:
- Parameterized SQL queries (SQL injection prevention)
- Input validation on all API endpoints
- Rate limiting support
- Audit logging capabilities
- RBAC policy interface

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve our security.
