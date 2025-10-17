# Public Repository Strategy Analysis

**Date**: 2025-10-14
**Prepared for**: Repository sync strategy decision

## Proposed Approaches

### Current Approach: Separate Working Directory
Internal repo (GitLab) → Copy/Redact → Separate working dir → GitHub

### Intern's Suggestion #1: Public Branch in Same Repo
Internal repo (GitLab) → `public` branch with redaction → Push to both GitLab + GitHub

### Intern's Suggestion #2: Move Everything to GitHub
Abandon internal GitLab → Use GitHub for everything → Public by default

## Detailed Analysis

---

## Option 1: Current Approach (Separate Working Directory)

**How it works**: Maintain two separate git repositories on disk, sync via scripts

### ✅ Pros

1. **Clear Separation**
   - Internal and public codebases are completely isolated
   - Zero risk of accidentally pushing internal refs to public
   - Different git histories can diverge if needed

2. **Flexible Redaction**
   - Can make different commits in public vs internal
   - Can maintain different documentation
   - Can have different contributor guidelines

3. **Safety**
   - Must explicitly run sync script to publish
   - Multiple review opportunities before public release
   - Easy to test redaction without affecting public repo

4. **Different Remotes**
   - Internal: GitLab (git@gitlab.iscinternal.com)
   - Public: GitHub (git@github.com:intersystems-community)
   - No risk of push confusion

### ❌ Cons

1. **Extra Disk Space**
   - Need ~500MB for second working directory
   - Doubled storage requirement

2. **Manual Sync Required**
   - Need to remember to run sync script
   - Can forget to sync after internal commits
   - Public repo can fall behind

3. **Complexity**
   - Need to maintain sync scripts
   - Two repos to manage
   - More cognitive overhead

4. **Merge Conflicts**
   - If public repo receives direct PRs
   - Need to sync changes back to internal

### 💰 Cost: Medium
- Disk: ~500MB
- Maintenance: Medium (sync scripts)
- Risk: Low (very safe)

---

## Option 2: Public Branch in Same Repo

**How it works**: Create a `public` branch in the same repo with redaction applied, push to both GitLab (internal) and GitHub (public)

### ✅ Pros

1. **Single Repository**
   - Only one working directory
   - Saves disk space
   - Simpler mental model

2. **Git-Native Solution**
   - Use git branches instead of separate repos
   - Standard git workflows
   - Easier for developers to understand

3. **Easier Syncing**
   - Git merge/rebase instead of rsync
   - Built-in conflict resolution
   - Track divergence with git

4. **Dual Push**
   - Can push to both GitLab and GitHub
   - Internal has full history
   - Public only sees redacted branch

### ❌ Cons

1. **Branch Maintenance Complexity**
   ```bash
   # Every commit requires:
   git checkout main              # Internal work
   git commit -m "feat: internal change"
   git checkout public            # Switch to public
   git merge main                 # Merge changes
   # Run redaction on changed files
   git commit --amend -m "chore: redact"
   git push github public:main    # Push to GitHub
   git push origin public         # Push to GitLab
   ```

2. **Risk of Mistakes**
   - Easy to accidentally push wrong branch
   - Could push `main` (internal) to GitHub by mistake
   - Git hooks required to prevent accidents

3. **Git History Confusion**
   - Public branch has "redaction commits" interspersed
   - Harder to see actual feature history
   - Merge commits create noise

4. **Redaction Timing**
   - When do you redact? After every commit?
   - Before merging to public branch?
   - Automated vs manual redaction?

5. **Branch Divergence**
   - `main` and `public` will diverge over time
   - Merge conflicts become common
   - Hard to track what's different

### Example Workflow

```bash
# Developer workflow
git checkout main
# ... work on feature ...
git commit -m "feat: add new pipeline"
git push origin main

# Sync to public (manual or automated)
git checkout public
git merge main --no-commit
python scripts/redact_for_public.py
git add -A
git commit -m "feat: add new pipeline (redacted)"
git push github public:main
```

### 💰 Cost: Medium-High
- Disk: 0MB (single repo)
- Maintenance: High (complex branching)
- Risk: Medium-High (easy to make mistakes)

### Required Safeguards

```bash
# Git hooks to prevent mistakes

# .git/hooks/pre-push
#!/bin/bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE=$1

if [[ "$REMOTE" == "github" && "$CURRENT_BRANCH" != "public" ]]; then
    echo "❌ ERROR: Cannot push $CURRENT_BRANCH to GitHub"
    echo "Only the 'public' branch can be pushed to GitHub"
    exit 1
fi

if [[ "$REMOTE" == "github" ]]; then
    # Verify no internal references
    if git diff origin/public..HEAD | grep -q "iscinternal"; then
        echo "❌ ERROR: Found internal references"
        exit 1
    fi
fi
```

---

## Option 3: Move Everything to GitHub

**How it works**: Abandon internal GitLab, use GitHub as primary repository, make everything public by default

### ✅ Pros

1. **Maximum Simplicity**
   - One repository, one source of truth
   - No sync needed
   - No redaction needed

2. **Community Engagement**
   - Public development from day one
   - External contributors can participate
   - Open source best practices

3. **GitHub Features**
   - Better CI/CD (GitHub Actions)
   - Better issue tracking
   - Better PR reviews
   - GitHub Copilot integration
   - Dependabot security updates

4. **No Maintenance**
   - No sync scripts to maintain
   - No redaction scripts needed
   - No branch management complexity

5. **Transparency**
   - Commit history is public
   - Development process visible
   - Builds trust with community

### ❌ Cons

1. **Internal References Problem**
   - Need to remove ALL internal references immediately
   - Docker: `docker.iscinternal.com` → Public registry
   - Emails: Company emails → Generic emails
   - Infrastructure: Internal URLs → Public URLs

2. **Sensitive Information Risk**
   - Must be extremely careful with commits
   - API keys, credentials must never be committed
   - Internal architecture details exposed
   - No "undo" if secrets leaked

3. **Organizational Policy**
   - May violate company policy
   - Legal review might be required
   - Compliance concerns (HIPAA, SOC2, etc.)
   - IP/Patent considerations

4. **Development Workflow Changes**
   - Can't commit WIP with TODO comments like "Ask John about this"
   - Can't reference internal tickets/systems
   - Must assume all commits are public

5. **Internal Testing**
   - Can't test against internal IRIS instances
   - Need public test infrastructure
   - Documentation must be self-contained

### Migration Requirements

If moving to GitHub entirely:

1. **Redact Entire Git History**
   ```bash
   # Use git-filter-repo to rewrite history
   git filter-repo --replace-text redaction-patterns.txt
   ```

2. **Update All Infrastructure References**
   - Docker images → Public registry
   - CI/CD → GitHub Actions
   - Documentation → Remove internal links

3. **Security Audit**
   - Scan entire history for secrets
   - Review all .env.example files
   - Check documentation for sensitive info

4. **Policy Approval**
   - Legal review
   - Security review
   - Management approval

### 💰 Cost: High (One-Time), Low (Ongoing)
- Migration: High (weeks of work)
- Maintenance: Zero (no sync)
- Risk: High (irreversible)

---

## Comparison Matrix

| Factor | Separate Repos | Public Branch | Full GitHub |
|--------|---------------|---------------|-------------|
| **Simplicity** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Maintenance** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Sync Ease** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (N/A) |
| **Disk Space** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Mistake Risk** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Community** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Setup Time** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Compliance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

---

## Recommendation

### Short Term (Current Sprint): **Separate Repos (Current Approach)**

**Why**:
- Already implemented and tested
- Safest option (zero risk of leaking internal refs)
- Flexible for different public/internal documentation
- Can evolve strategy later

**Action**: Continue using current sync scripts

### Medium Term (Next Quarter): **Evaluate Public Branch Approach**

**Why**:
- Once sync workflows are established
- Can test with feature branches first
- Assess if branch complexity is manageable

**Action**: Create proof-of-concept public branch

### Long Term (6-12 months): **Consider Full GitHub Migration**

**Why**:
- If repository is truly open source
- If company policy allows
- If internal references are fully removed
- If public development is the goal

**Action**: Conduct feasibility study

---

## Hybrid Approach (Recommended)

Combine benefits of all three:

### Phase 1: Current (Separate Repos) ✅
- Use separate working directory
- Automated sync scripts (already built)
- Safe, predictable

### Phase 2: Public Branch Experiment (2-3 months)
- Create `public` branch in internal repo
- Test dual-push workflow
- Evaluate developer experience
- Keep separate repos as backup

### Phase 3: Decision Point (6 months)
- If public branch works well → Deprecate separate repos
- If too complex → Stay with separate repos
- If fully open source → Migrate to GitHub entirely

### Example Phase 2 Workflow

```bash
# Setup (one time)
git checkout -b public main
python scripts/redact_for_public.py
git commit -m "chore: create redacted public branch"
git remote add github git@github.com:intersystems-community/iris-rag-templates.git
git push github public:main

# Daily workflow (automated with script)
./scripts/sync_public_branch.sh  # New script that:
# 1. Merges main → public
# 2. Runs redaction
# 3. Commits
# 4. Pushes to both GitLab and GitHub
```

---

## Decision Criteria

### Choose Separate Repos If:
- ✅ Need maximum safety
- ✅ Have different documentation for internal/public
- ✅ Want flexibility to diverge
- ✅ Don't mind extra disk space
- ✅ Want explicit control over what goes public

### Choose Public Branch If:
- ✅ Want git-native solution
- ✅ Developers comfortable with branching
- ✅ Have good git hooks/automation
- ✅ Want to save disk space
- ✅ Internal and public are mostly the same

### Choose Full GitHub If:
- ✅ Repository is truly open source
- ✅ No sensitive internal infrastructure
- ✅ Company policy allows
- ✅ Want maximum community engagement
- ✅ No internal-only features

---

## Action Items

### Immediate (This Week)
1. ✅ Continue with separate repos (current approach)
2. Document current sync workflow
3. Get feedback from team on sync process

### Short Term (Next Month)
1. Create proof-of-concept public branch
2. Test dual-push workflow with one feature
3. Compare developer experience

### Medium Term (Next Quarter)
1. Evaluate public branch success
2. Make decision: stay separate or switch to public branch
3. Update documentation based on decision

### Long Term (6+ Months)
1. Assess if full GitHub migration is feasible
2. Conduct security/compliance review
3. Plan migration if approved

---

## Conclusion

**Current Recommendation**: **Stick with Separate Repos**

**Reasoning**:
1. ✅ Already implemented and working
2. ✅ Safest approach (zero risk)
3. ✅ Most flexible for future changes
4. ✅ Can evolve to other approaches later
5. ✅ Only ~500MB disk space cost

**However**: Worth experimenting with public branch as Phase 2

The intern's suggestions are valid! The public branch approach is cleaner architecturally, but comes with complexity and risk. The full GitHub migration is ideal for true open source, but requires organizational buy-in.

**Best Path Forward**:
1. Use separate repos now (working, safe)
2. Experiment with public branch next quarter
3. Evaluate GitHub migration in 6+ months

This gives you:
- Immediate safety ✅
- Future flexibility ✅
- Data-driven decision making ✅

---

**Questions to Consider**:

1. How often do you need to sync? (Daily? Weekly?)
2. Do you expect external contributors to the public repo?
3. Are there truly internal-only features that shouldn't be public?
4. What's the company policy on open source development?
5. How comfortable is the team with complex git workflows?

**Want to discuss any specific approach in more detail?**
