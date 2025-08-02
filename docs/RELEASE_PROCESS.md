# Release Process Guide

This document outlines the professional release process for the RAG Templates project.

## Versioning Strategy

### Semantic Versioning
We follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): Backwards-compatible functionality additions
- **PATCH** (0.0.X): Backwards-compatible bug fixes

### Pre-release Versions
- **Alpha**: `X.Y.Z-alpha.N` - Early development, may be unstable
- **Beta**: `X.Y.Z-beta.N` - Feature complete, testing in progress  
- **Release Candidate**: `X.Y.Z-rc.N` - Final testing before release

### Development Versions
- **Development**: `X.Y.Z-dev.N` - Ongoing development snapshots

## Release Checklist

### Pre-Release (1-2 weeks before)
- [ ] Feature freeze - no new features, only bug fixes
- [ ] Update documentation for all new features
- [ ] Run comprehensive test suite (`make test-ragas-1000-enhanced`)
- [ ] Performance benchmarking and regression testing
- [ ] Security review and dependency updates

### Release Preparation (1 week before)
- [ ] Update CHANGELOG.md with all changes since last release
- [ ] Create release highlights document
- [ ] Update version in pyproject.toml
- [ ] Update any version references in documentation
- [ ] Create migration guide if breaking changes exist

### Release Day
- [ ] Final test run on clean environment
- [ ] Create and push version tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
- [ ] Sync to public repository: `python scripts/sync_to_public.py --sync-all --push`
- [ ] Create GitHub release with highlights and binaries
- [ ] Publish to PyPI (if applicable)
- [ ] Update InterSystems Open Exchange listing

### Post-Release (within 1 week)
- [ ] Monitor for critical issues and feedback
- [ ] Update documentation site
- [ ] Announce on relevant channels (InterSystems Developer Community, etc.)
- [ ] Plan next release milestone

## Release Types

### Major Release (X.0.0)
**Triggers:**
- Breaking API changes
- Major architecture changes
- New core functionality that changes user workflows

**Timeline:** 3-6 months
**Example:** v1.0.0, v2.0.0

### Minor Release (0.X.0)
**Triggers:**
- New RAG techniques
- New features that don't break existing API
- Significant performance improvements
- New integration capabilities

**Timeline:** 1-2 months  
**Example:** v0.2.0 (current), v0.3.0

### Patch Release (0.0.X)
**Triggers:**
- Bug fixes
- Security updates
- Documentation improvements
- Minor performance optimizations

**Timeline:** As needed (1-2 weeks)
**Example:** v0.2.1, v0.2.2

## Version Management

### Current Version: v0.2.0
This major minor release includes:
- Requirements-driven orchestrator architecture
- Unified Query() API
- Basic reranking pipeline
- Critical infrastructure fixes

### Next Planned: v0.3.0
Tentative features:
- Advanced RAG techniques (RAG-Fusion, Self-RAG)
- Multi-modal document processing
- Enhanced performance optimizations
- Enterprise deployment guides

## Release Automation

### Git Workflow
```bash
# Create release branch
git checkout -b release/v0.2.0

# Update version and changelog
# ... make changes ...

# Commit release changes
git commit -m "chore: prepare release v0.2.0"

# Create tag
git tag -a v0.2.0 -m "Release v0.2.0: Enterprise RAG Architecture Milestone"

# Merge to main
git checkout main
git merge release/v0.2.0

# Push tag
git push origin v0.2.0
git push origin main
```

### Public Sync
```bash
# Sync to public repository
python scripts/sync_to_public.py --sync-all --push
```

### GitHub Release
1. Go to GitHub repository releases
2. Click "Create a new release"
3. Select the version tag (v0.2.0)
4. Use release highlights as description
5. Attach any relevant binaries or documentation

## Quality Gates

Before any release, the following must pass:

### Automated Tests
- [ ] Unit tests: `make test-unit`
- [ ] Integration tests: `make test-integration`  
- [ ] E2E tests: `make test-e2e`
- [ ] 1000-doc validation: `make test-1000`
- [ ] RAGAS evaluation: `make test-ragas-1000-enhanced`

### Code Quality
- [ ] Linting: `make lint`
- [ ] Type checking: `uv run mypy iris_rag/`
- [ ] Security scan: `safety check`
- [ ] Dependency audit: `pip-audit`

### Documentation
- [ ] All new features documented
- [ ] API documentation updated
- [ ] Migration guide (if breaking changes)
- [ ] Release highlights completed

### Performance
- [ ] Benchmark results within acceptable ranges
- [ ] Memory usage profiling
- [ ] Load testing for high-volume scenarios

## Communication

### Internal Communication
- Update project stakeholders via GitLab issues
- Post release notes in internal documentation
- Schedule release review meetings

### External Communication
- GitHub release announcement
- InterSystems Developer Community post
- Update project README and documentation site
- Social media announcements (if applicable)

---

This process ensures professional, reliable releases that meet enterprise standards while maintaining development velocity.