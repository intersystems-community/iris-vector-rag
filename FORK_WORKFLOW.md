# Fork Workflow Strategy

## Overview
Development happens on private fork, PRs go to public repo. Private development artifacts stay in fork only.

## Git Remote Setup

### Current State
```bash
github → https://github.com/intersystems-community/iris-vector-rag.git (public repo)
```

### Recommended Setup
```bash
# After creating your private fork on GitHub (e.g., yourusername/iris-vector-rag-private)

# 1. Rename current remote to 'upstream' (public repo)
git remote rename github upstream

# 2. Add your private fork as 'origin'
git remote add origin https://github.com/YOURUSERNAME/iris-vector-rag-private.git

# 3. Verify remotes
git remote -v
# Should show:
# origin    https://github.com/YOURUSERNAME/iris-vector-rag-private.git (fetch)
# origin    https://github.com/YOURUSERNAME/iris-vector-rag-private.git (push)
# upstream  https://github.com/intersystems-community/iris-vector-rag.git (fetch)
# upstream  https://github.com/intersystems-community/iris-vector-rag.git (push)
```

## Branch Strategy

### Development Branches (on private fork)
- `main` - Your main development branch (includes all private files)
- `feature/<feature-name>` - Feature branches with private artifacts
- `public-main` - Clean branch for public repo (no private files)
- `pr/<feature-name>` - Branches prepared for PRs (cleaned of private files)

### Public Repo Branches (upstream)
- `main` - Public main branch (clean, no private files)

## Private Files Configuration

### Update .gitignore for Private Fork
Add to `.gitignore` (these files won't be in public PRs):
```gitignore
# Private development files (not for public repo)
.claude/
.specify/
STATUS.md
PROGRESS.md
TODO.md
FORK_WORKFLOW.md

# Keep specs/ if you want them private, or remove this line if public
# specs/
```

### Current Tracked Private Files
These are currently tracked in git and need to be removed from public PRs:
- `.claude/` - 7 command files
- `.specify/` - Config, constitution, scripts, templates
- Various tracking files in `docs/` and `contrib/`

## Workflow: Private Development → Public PR

### Step 1: Daily Development (Private Fork)
```bash
# Work on private fork main branch
git checkout main
# ... make changes ...
git add .
git commit -m "feat: implement new feature (with private notes)"
git push origin main
```

### Step 2: Prepare PR Branch (Clean for Public)
```bash
# Create PR branch from your private work
git checkout -b pr/feature-051-embedding main

# Remove private files from this branch only
git rm -r --cached .claude/
git rm -r --cached .specify/
git rm --cached STATUS.md PROGRESS.md TODO.md
git rm --cached FORK_WORKFLOW.md

# Commit the removal
git commit -m "chore: remove private development files for public PR"

# Push to your private fork
git push origin pr/feature-051-embedding
```

### Step 3: Create Pull Request
1. Go to GitHub: `https://github.com/intersystems-community/iris-vector-rag`
2. Click "New Pull Request"
3. Click "compare across forks"
4. Set:
   - **Base repository**: `intersystems-community/iris-vector-rag` (base: `main`)
   - **Head repository**: `YOURUSERNAME/iris-vector-rag-private` (compare: `pr/feature-051-embedding`)
5. Create PR with clean code only (no private files)

### Step 4: After PR Merged (Sync Back)
```bash
# Update your private fork from public repo
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

## Alternative: Use git-filter-repo for Automatic Cleaning

If you want to automate PR branch creation, use `git-filter-repo`:

```bash
# Install git-filter-repo
brew install git-filter-repo  # macOS
# or: pip install git-filter-repo

# Create clean PR branch automatically
git checkout -b pr/feature-051-embedding main
git filter-repo --invert-paths \
  --path .claude/ \
  --path .specify/ \
  --path STATUS.md \
  --path PROGRESS.md \
  --path TODO.md \
  --path FORK_WORKFLOW.md \
  --force

git push origin pr/feature-051-embedding
```

**WARNING**: `git-filter-repo` rewrites history. Only use on PR branches, NEVER on main!

## Sync Strategy

### Keep Private Fork Updated with Public Repo
```bash
# Fetch public repo changes
git fetch upstream

# Merge public changes into your private main
git checkout main
git merge upstream/main

# Resolve any conflicts (private files won't conflict)
git push origin main
```

### Regular Sync Schedule
- Daily: Push private development to `origin main`
- Before PR: Create clean PR branch
- After PR merge: Sync `upstream/main` → `origin/main`

## Current State Migration

### Option A: Start Fresh with Private Files
1. Create private fork on GitHub
2. Clone to new directory
3. Keep `.claude/` and `.specify/` in private fork
4. Update public repo's .gitignore to exclude these
5. Future PRs won't include private files

### Option B: Retroactively Clean Public Repo (Dangerous!)
1. Use `git-filter-repo` to remove private files from public repo history
2. Force push to public repo (requires admin access)
3. **WARNING**: This breaks all existing clones!

**RECOMMENDATION**: Use Option A. Keep existing public repo history as-is, just exclude private files going forward.

## Questions to Decide

1. **Fork Location**: Where will you create the private fork?
   - GitHub private repo?
   - GitLab private repo?
   - Enterprise GitHub instance?

2. **Specs Directory**: Should `specs/` be private or public?
   - Private: Feature planning stays internal
   - Public: Transparent development process

3. **Constitution**: Should `.specify/memory/constitution.md` be public?
   - Could be valuable for community understanding development principles
   - Or keep private as internal process documentation

4. **Current Tracking Files**: What to do with tracked files in `docs/`?
   - Remove from public repo?
   - Keep as historical documentation?

## Next Steps

1. Create private fork on GitHub (if not already created)
2. Update .gitignore in public repo to exclude private files
3. Set up git remotes (origin = private, upstream = public)
4. Decide on specs/ and constitution visibility
5. Test workflow with a small PR
6. Document your specific workflow preferences

## References
- Git Filter Repo: https://github.com/newren/git-filter-repo
- GitHub Forking Workflow: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks
- Git Remote Management: https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes
