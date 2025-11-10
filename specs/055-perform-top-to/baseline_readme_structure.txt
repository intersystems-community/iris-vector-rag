============================= test session starts ==============================
platform darwin -- Python 3.12.9, pytest-8.4.1, pluggy-1.6.0 -- /opt/homebrew/Caskroom/miniconda/base/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
metadata: {'Python': '3.12.9', 'Platform': 'macOS-15.5-arm64-arm-64bit', 'Packages': {'pytest': '8.4.1', 'pluggy': '1.6.0'}, 'Plugins': {'docker': '3.2.3', 'langsmith': '0.4.41', 'mock': '3.15.1', 'asyncio': '1.2.0', 'hypothesis': '6.139.2', 'anyio': '4.11.0', 'html': '4.1.1', 'env': '1.1.5', 'xdist': '3.8.0', 'timeout': '2.4.0', 'metadata': '3.1.1', 'locust': '2.40.5', 'clarity': '1.0.1', 'order': '1.3.0', 'pytest_httpserver': '1.1.3', 'redis': '3.1.3', 'sugar': '1.1.1', 'benchmark': '5.1.0', 'asyncio-cooperative': '0.40.0', 'Faker': '37.12.0', 'cov': '6.1.1', 'postgresql': '7.0.2'}, 'JAVA_HOME': '/Library/Java/JavaVirtualMachines/temurin-21.jdk/Contents/Home'}
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /Users/tdyar/ws/iris-vector-rag-private
configfile: pytest.ini
plugins: docker-3.2.3, langsmith-0.4.41, mock-3.15.1, asyncio-1.2.0, hypothesis-6.139.2, anyio-4.11.0, html-4.1.1, env-1.1.5, xdist-3.8.0, timeout-2.4.0, metadata-3.1.1, locust-2.40.5, clarity-1.0.1, order-1.3.0, pytest_httpserver-1.1.3, redis-3.1.3, sugar-1.1.1, benchmark-5.1.0, asyncio-cooperative-0.40.0, Faker-37.12.0, cov-6.1.1, postgresql-7.0.2
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 7 items

specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_line_count_under_400 FAILED [ 14%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_has_value_proposition_in_first_paragraph FAILED [ 28%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_has_clear_heading_structure PASSED [ 42%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_links_to_detailed_documentation PASSED [ 57%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_quick_start_is_complete FAILED [ 71%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_uses_professional_language PASSED [ 85%]
specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_has_documentation_section PASSED [100%]

=================================== FAILURES ===================================
_______________________ test_readme_line_count_under_400 _______________________
specs/055-perform-top-to/contracts/readme_structure_contract.py:97: in test_readme_line_count_under_400
    assert line_count <= 400, \
E   AssertionError: README.md has 518 lines (exceeds 400-line limit by 118 lines)
E   assert 518 <= 400
_____________ test_readme_has_value_proposition_in_first_paragraph _____________
specs/055-perform-top-to/contracts/readme_structure_contract.py:115: in test_readme_has_value_proposition_in_first_paragraph
    assert len(first_para) >= 50, \
E   AssertionError: First paragraph too short (0 chars) - should communicate value proposition
E   assert 0 >= 50
E    +  where 0 = len('')
_____________________ test_readme_quick_start_is_complete ______________________
specs/055-perform-top-to/contracts/readme_structure_contract.py:192: in test_readme_quick_start_is_complete
    assert len(missing_elements) == 0, \
E   AssertionError: Quick Start section incomplete - missing: ['install', 'python', 'create_pipeline']
E   assert 3 == 0
E    +  where 3 = len(['install', 'python', 'create_pipeline'])
=============================== warnings summary ===============================
specs/055-perform-top-to/contracts/readme_structure_contract.py:91
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:91: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:101
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:101: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:122
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:122: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:148
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:148: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:170
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:170: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:196
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:196: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

specs/055-perform-top-to/contracts/readme_structure_contract.py:228
  /Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/contracts/readme_structure_contract.py:228: PytestUnknownMarkWarning: Unknown pytest.mark.contract - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.contract

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_line_count_under_400
FAILED specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_has_value_proposition_in_first_paragraph
FAILED specs/055-perform-top-to/contracts/readme_structure_contract.py::test_readme_quick_start_is_complete
=================== 3 failed, 4 passed, 7 warnings in 0.17s ====================
