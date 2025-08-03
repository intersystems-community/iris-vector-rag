# Current TODO List

## High Priority
- [x] Fix ObjectScript compilation errors in public GitHub repository
- [x] Analyze and fix RAG.SourceDocumentsWithIFind class syntax errors  
- [x] Fix RAG.IFindSetup table reference and method entry point issues
- [x] Run IPM/ZPM validator script before syncing
- [x] Test ZPM package compilation after fixes
- [x] Sync fixes to public GitHub repository ✅ COMPLETED
- [🔄] Fix ZPM dependency version in module.xml

## Medium Priority

## Completed Fixes
- Removed SQL statements referencing tables during compilation from SourceDocumentsWithIFind.cls
- Added default parameter to TestIFindSearch method in IFindSetup.cls
- Removed circular table dependencies between IFindSetup and SourceDocumentsWithIFind
- Added error checking for source table existence before copy operations

## Important Reminders
- **ObjectScript files must use .CLS extension** (not .cls)
- Always check file extensions when working with IRIS ObjectScript
- Use uppercase .CLS for proper IRIS compilation
- **ALWAYS run IPM validator script before pushing to public GitHub**
  - `python scripts/validate_ipm_package.py .`
  - `python scripts/utilities/validate_ipm_module.py --project-root .`
  - Both must pass ✅ before syncing

## Recent Commands
1. `Edit module.xml` - Fixed ZPM dependency (removed %ZPM reference, kept empty Dependencies)
2. `python scripts/utilities/validate_ipm_module.py --project-root .` - Validation PASSED with empty Dependencies
3. `git push github community-edition-defaults` - ZPM dependency fix pushed to GitHub
4. **Research**: ZPM packages typically don't declare %ZPM itself as dependency
5. **🔧 ZPM DEPENDENCY ISSUE RESOLVED** - Empty Dependencies section is correct pattern

---
*Last updated: 2025-08-03*