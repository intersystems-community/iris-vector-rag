# RAG-Templates TODO

**Last Updated**: 2025-10-05

## Immediate (This Session)

### Vector Store - DONE ✅
- [x] Fix password reset infinite loop
- [x] Fix schema column mismatches
- [x] Fix embedding generation
- [x] Implement similarity_search_with_score
- [x] Mark JSON filtering tests as xfail

### Next Steps
- [ ] GraphRAG E2E tests (~49 failures to investigate)
- [ ] Coverage improvements (current: 10%, target: 95%)
- [ ] Feature 028 completion

## Short Term (This Week)

### Testing
- [ ] Investigate GraphRAG E2E test failures
- [ ] Add more unit tests for coverage
- [ ] Contract tests for all pipelines

### Infrastructure  
- [ ] IRIS JSON metadata filtering (research needed)
- [ ] Test data cleanup automation
- [ ] Pre-flight check improvements

## Medium Term (Next 2 Weeks)

### iris-devtools Package
- [ ] Extract connection management to iris-devtools
- [ ] Extract password reset utilities
- [ ] Extract testcontainers wrapper
- [ ] Publish to PyPI
- [ ] Migrate rag-templates to use iris-devtools

### Documentation
- [ ] Update architecture docs
- [ ] Document IRIS JSON limitations
- [ ] Add troubleshooting guide

### Performance
- [ ] GraphRAG optimization
- [ ] Query performance benchmarking
- [ ] Memory usage profiling

## Long Term (This Month)

### Production Readiness
- [ ] 95%+ test coverage
- [ ] All E2E tests passing (or xfailed with reasons)
- [ ] Performance benchmarks established
- [ ] Documentation complete

### New Features
- [ ] Advanced metadata filtering (IRIS-specific)
- [ ] Query optimization
- [ ] Hybrid search improvements

## Deferred (Future)

### Nice to Have
- [ ] Multi-pipeline comparison UI
- [ ] Advanced visualization
- [ ] Real-time monitoring dashboard
- [ ] A/B testing framework

## Known Issues

### Critical
None currently

### High Priority
- IRIS JSON metadata filtering (5 xfailed tests)
- GraphRAG E2E test failures
- Low test coverage (10%)

### Medium Priority
- Performance optimization needed
- Documentation gaps

### Low Priority  
- Minor UI improvements
- Additional pipeline variants

## Notes
- Focus on test stability and coverage
- iris-devtools will improve reusability
- Document all IRIS-specific learnings
