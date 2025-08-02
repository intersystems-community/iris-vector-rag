# Repository Synchronization

This document describes the automated repository synchronization system that keeps documentation and selected files synchronized between the internal GitLab repository and the public GitHub repository.

## Overview

The repository synchronization system automates the process of:

1. **Documentation Synchronization**: Copying updated README files from the sanitized public repository to the internal repository
2. **Source Code Synchronization**: Syncing core source code while filtering out internal/private content
3. **Git Operations**: Staging, committing, and pushing changes to the internal GitLab repository
4. **Validation**: Checking synchronization status and ensuring files are up-to-date

## Quick Start

### Using Makefile (Recommended)

```bash
# Repository Synchronization
make sync-dry-run          # Preview synchronization (dry run)
make sync-docs             # Synchronize documentation files only
make sync-docs-push        # Synchronize documentation and push to GitLab

make sync-all-dry-run      # Preview comprehensive sync (dry run)
make sync-all              # Synchronize all content (docs + source code)
make sync-all-push         # Synchronize all content and push to GitLab

# Status Check
make sync-check            # Check synchronization status
```

### Using Script Directly

```bash
# Documentation synchronization
python scripts/sync_repositories.py --sync-docs
python scripts/sync_repositories.py --sync-docs --push

# Comprehensive synchronization
python scripts/sync_repositories.py --sync-all
python scripts/sync_repositories.py --sync-all --push

# Validation and dry runs
python scripts/sync_repositories.py --validate-sync
python scripts/sync_repositories.py --sync-all --dry-run
```

## Configuration

The synchronization behavior is controlled by [`config/sync_config.yaml`](../config/sync_config.yaml):

```yaml
# Repository paths
repositories:
  internal_repo_path: "."
  sanitized_repo_path: "../rag-templates-sanitized"

# Git configuration
git:
  branch: "feature/enterprise-rag-system-complete"
  commit_message_template: "docs: sync documentation updates from sanitized repository"

# Files to synchronize
files_to_sync:
  - source: "README.md"
    target: "README.md"
    description: "Main project README"
  
  - source: "docs/README.md"
    target: "docs/README.md"
    description: "Documentation directory README"
  
  - source: "rag_templates/README.md"
    target: "rag_templates/README.md"
    description: "RAG templates module README"
```

## Architecture

### Components

1. **`scripts/sync_repositories.py`**: Unified synchronization script supporting both documentation-only and comprehensive sync
2. **`config/sync_config.yaml`**: Configuration file with directory sync support
3. **Makefile targets**: Convenient command aliases for sync operations

### Classes

- **`SyncConfig`**: Configuration data structure
- **`SyncResult`**: Result tracking for operations
- **`RepositorySynchronizer`**: Main synchronization logic

### Key Features

- **YAML Configuration**: Flexible, version-controlled configuration with directory sync support
- **Content Filtering**: Intelligent filtering to exclude internal/private content from public sync
- **Directory Synchronization**: Comprehensive directory-level sync with pattern matching
- **Dry Run Mode**: Preview changes without applying them
- **Validation**: Check synchronization status across all content types
- **Error Handling**: Comprehensive error reporting and recovery
- **Git Integration**: Automatic staging, committing, and pushing

## Workflow

### Manual Synchronization Process

The script automates what was previously done manually:

1. **Copy Files**: Copy updated documentation from sanitized repository
2. **Stage Changes**: `git add` modified files
3. **Commit**: Create commit with descriptive message
4. **Push**: Push to GitLab repository (optional)

### Automated Validation

The script can validate synchronization status:

- Compare file contents between repositories
- Report sync percentage
- Identify missing or out-of-sync files

## Usage Examples

### Development Workflow

```bash
# After updating documentation in sanitized repository
make sync-dry-run          # Preview changes
make sync-docs             # Apply changes locally
make sync-docs-push        # Apply and push to GitLab
```

### CI/CD Integration

```bash
# Check if sync is needed (exit code 1 if changes needed)
make sync-check

# Automated sync in CI pipeline
make sync-docs-push
```

### Custom Configuration

```bash
# Use custom configuration file
python scripts/sync_repositories.py --config-file custom_sync.yaml --sync-docs
```

## File Structure

```
├── scripts/
│   └── sync_repositories.py          # Unified sync script (docs + source code)
├── config/
│   └── sync_config.yaml              # Configuration with directory sync
├── docs/
│   └── REPOSITORY_SYNC.md            # This documentation
└── Makefile                          # Convenient targets for sync operations
```

## Exit Codes

- **0**: Success, no changes needed or operation completed successfully
- **1**: Changes needed (for validation) or operation failed

## Error Handling

The script handles various error conditions:

- **Missing repositories**: Clear error if paths don't exist
- **Git failures**: Detailed error messages for git operations
- **File access issues**: Proper error reporting for file operations
- **Configuration errors**: Validation of YAML configuration

## Security Considerations

- **No secrets**: Configuration files contain no sensitive information
- **Path validation**: Repository paths are validated before operations
- **Git safety**: Uses standard git commands with proper error handling

## Troubleshooting

### Common Issues

1. **Repository not found**
   ```
   Error: Sanitized repository path does not exist: ../rag-templates-sanitized
   ```
   **Solution**: Ensure the sanitized repository is cloned in the expected location

2. **Git operation failed**
   ```
   Git operation failed: fatal: not a git repository
   ```
   **Solution**: Ensure you're running from within the git repository

3. **Permission denied**
   ```
   Permission denied: config/sync_config.yaml
   ```
   **Solution**: Check file permissions and ensure you have write access

### Debug Mode

For detailed logging, modify the script's logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements to the synchronization system:

1. **Bidirectional Sync**: Support syncing changes back to sanitized repository
2. **Conflict Resolution**: Advanced merge strategies for conflicting changes
3. **Webhook Integration**: Automatic triggering on repository updates
4. **Multiple Branches**: Support for syncing across different branches
5. **File Filtering**: More sophisticated file selection rules

## Related Documentation

- [Main README](../README.md): Project overview
- [Development Guide](../docs/README.md): Development documentation
- [RAG Templates Guide](../rag_templates/README.md): Module documentation