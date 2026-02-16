# iris-vector-rag-private Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-03

## Active Technologies
- Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for HybridGraphRAG) (060-fix-users-tdyar)
- InterSystems IRIS (vector database with SQL interface) (060-fix-users-tdyar)
- Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for GraphRAG) (060-fix-users-tdyar)

- Python 3.12, Docker, GitHub Actions (Ubuntu 24.04) + Checkov, Docker, GitHub Actions (001-fix-ci-security-failures)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.12, Docker, GitHub Actions (Ubuntu 24.04): Follow standard conventions

## Recent Changes
- 060-fix-users-tdyar: Added Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for GraphRAG)
- 060-fix-users-tdyar: Added Python 3.12 + intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for HybridGraphRAG)

- 001-fix-ci-security-failures: Added Python 3.12, Docker, GitHub Actions (Ubuntu 24.04) + Checkov, Docker, GitHub Actions

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
---

## COMPOUND KNOWLEDGE BASE & DEVELOPMENT ENVIRONMENT

**Last Updated**: $(date +%Y-%m-%d)

> Central registry of development knowledge, tools, capabilities, and automation


### 📚 Knowledge Base

**Location**: `~/.config/opencode/compound-knowledge/`


**Structure**:
- `global/` - Project-agnostic knowledge (frameworks, tools, patterns)
- `projects/[name]/` - Project-specific knowledge (domain, architecture)

**Stats**: 7 solutions (7 global, 0 project)

**Quick Search**:
```bash
# Search everything
grep -ri "keywords" ~/.config/opencode/compound-knowledge/

# Global only
grep -ri "keywords" ~/.config/opencode/compound-knowledge/global/

# This project
grep -ri "keywords" ~/.config/opencode/compound-knowledge/projects/$(basename $(pwd))/
```

**Time Savings**: First solve: 30min → Next solve: 3min (90% faster)


### 🔌 MCP Servers

- `atlassian` - local
- `gemini-impl` - local
- `hallucination-detector` - local
- `jama` - local
- `perplexity` - local
- `playwright` - local
- `qwen-impl` - local
- `support-tools` - local

### 🤖 Automation & Hooks

**Orchestrator Behavior** (configured via `~/.config/opencode/oh-my-opencode-slim.json`):
- **Before tasks**: Search compound KB for similar solutions
- **After completion**: Remind to document solution

**Periodic Maintenance** (recommended):
```bash
# Weekly: Regenerate KB index
~/.config/opencode/compound-knowledge/generate-index.sh

# Monthly: Review and consolidate similar solutions
# Quarterly: Extract patterns from repeated solutions
```

**Auto-Documentation Triggers**:
- Tests pass after fixing failure → Document solution
- Error resolved → Document fix
- Performance improved → Document optimization
- Integration working → Document configuration


### 🛠️ Tools & Utilities

**Compound Engineering**:
- `~/.config/opencode/compound-knowledge/new-solution.sh` - Create solution doc
- `~/.config/opencode/compound-knowledge/generate-index.sh` - Regenerate index
- `~/.config/opencode/compound-knowledge/sync-to-agents.sh` - Update AGENTS.md files

**OpenCode Agents** (oh-my-opencode-slim):
- `orchestrator` - Master coordinator (with compound engineering)
- `explorer` - Codebase reconnaissance
- `oracle` - Strategic advisor
- `librarian` - External knowledge (websearch, context7, grep_app MCPs)
- `designer` - UI/UX implementation
- `fixer` - Fast implementation
- `code-simplifier` - Post-work code refinement


### 🎯 Skills

**speckit** (feature specification):
- `speckit.plan` - Implementation planning
- `speckit.specify` - Feature specification
- `speckit.tasks` - Task generation
- `speckit.implement` - Implementation guidance


### 📋 Quick Reference

**Full Index**: `~/.config/opencode/compound-knowledge/INDEX.md`

**Documentation Templates**:
```markdown
---
title: "Problem description"
category: [build-errors|test-failures|runtime-errors|performance-issues|
          database-issues|security-issues|ui-bugs|integration-issues|logic-errors]
date: YYYY-MM-DD
severity: high|medium|low
tags: [tag1, tag2]
time_to_solve: XXmin
---

## Problem Symptom
[What was observed]

## Solution
[How you fixed it]

## Prevention
[How to avoid future]
```

**Decision Tree** (Global vs Project-Specific):
- Framework/tool issue → `global/`
- General pattern → `global/`
- Project domain logic → `projects/[name]/`
- Project architecture → `projects/[name]/`
- When in doubt → `global/`

---
