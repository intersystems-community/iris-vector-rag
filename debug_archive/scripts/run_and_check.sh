#!/bin/bash
# Run tests and check if they're actually executing (not skipped)

# Run the tests with real embeddings
./run_tests_with_1000_docs.sh graphrag > test_output.log 2>&1

# Check test results
echo "Analyzing test results..."
echo ""

# Count total tests, passed tests, and skipped tests
TOTAL=$(grep "collected" test_output.log | grep -oE '[0-9]+' | head -1)
PASSED=$(grep "PASSED" test_output.log | wc -l)
SKIPPED=$(grep "SKIPPED" test_output.log | wc -l)
FAILED=$(grep "FAILED" test_output.log | wc -l)

# Show summary
echo "Test Summary:"
echo "  Total tests: $TOTAL"
echo "  Passed: $PASSED"
echo "  Skipped: $SKIPPED"
echo "  Failed: $FAILED"
echo ""

# Check if any tests actually ran
if [ "$PASSED" -gt 0 ]; then
  echo "SUCCESS: Tests are running and passing!"
  grep "PASSED" test_output.log
elif [ "$FAILED" -gt 0 ]; then
  echo "PARTIAL SUCCESS: Tests are running but some are failing"
  grep "FAILED" test_output.log -A 5
else
  echo "FAILURE: All tests are being skipped or not running"
  grep "SKIPPED" test_output.log -A 2
fi

# Show excerpts from the log file
echo ""
echo "Excerpts from log file:"
grep -n "Loading PMC documents" test_output.log -A 3 || echo "No document loading occurred"
grep -n "Building knowledge graph" test_output.log -A 3 || echo "No knowledge graph building occurred"
grep -n "Running query" test_output.log -A 3 || echo "No queries were executed"

echo ""
echo "For full details, see test_output.log"
