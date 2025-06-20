# Testing Framework Integration Validation Report

**Generated:** 2025-06-15T13:50:36.048537
**Duration:** 0.36 seconds
**Status:** None

## Component Validation Results

### Existence Check

- ✅ **test_modes**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/test_modes.py', 'size': 4129}
- ✅ **conftest_test_modes**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/conftest_test_modes.py', 'size': 3258}
- ✅ **post_installation_tests**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/scripts/run_post_installation_tests.py', 'size': 10890}
- ✅ **e2e_validation**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/scripts/run_e2e_validation.py', 'size': 24564}
- ✅ **cross_language_integration**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/test_cross_language_integration.py', 'size': 23750}
- ✅ **real_data_validation**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/test_real_data_validation.py', 'size': 52120}
- ✅ **mode_validator**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/test_mode_validator.py', 'size': 13938}
- ✅ **main_conftest**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/tests/conftest.py', 'size': 28630}
- ✅ **makefile**: {'exists': True, 'path': '/Users/tdyar/ws/rag-templates/Makefile', 'size': 32430}

### Test Mode Framework

- ✅ **import_test_modes**: True
- ✅ **mode_switching**: True
- ✅ **mock_control**: True

### Mock Control System

- ❌ **tests_executed**: False
- ✅ **stdout**: [1m============================= test session starts ==============================[0m
platform darwin -- Python 3.12.9, pytest-7.4.4, pluggy-1.6.0 -- /Users/tdyar/ws/rag-templates/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/tdyar/ws/rag-templates
configfile: pytest.ini
plugins: anyio-4.9.0, cov-4.1.0, mock-3.14.0, lazy-fixture-0.6.3
[1mcollecting ... [0mcollected 24 items

tests/test_mode_validator.py::TestMockController::test_default_mode_is_unit 
[1m-------------------------------- live log setup --------------------------------[0m
2025-06-15 13:50:36 [[32m    INFO[0m] 🧪 Test Mode: E2E (conftest.py:65)
2025-06-15 13:50:36 [[32m    INFO[0m] 🎭 Mocks Disabled: True (conftest.py:66)
2025-06-15 13:50:36 [[32m    INFO[0m] 🗄️  Real Database Required: True (conftest.py:67)
2025-06-15 13:50:36 [[32m    INFO[0m] 📊 Real Data Required: True (conftest.py:68)
[32mPASSED[0m[32m                                                                   [  4%][0m
tests/test_mode_validator.py::TestMockController::test_set_unit_mode [32mPASSED[0m[32m [  8%][0m
tests/test_mode_validator.py::TestMockController::test_set_integration_mode [32mPASSED[0m[32m [ 12%][0m
tests/test_mode_validator.py::TestMockController::test_set_e2e_mode [32mPASSED[0m[32m [ 16%][0m
tests/test_mode_validator.py::TestMockController::test_environment_variable_detection [32mPASSED[0m[32m [ 20%][0m
tests/test_mode_validator.py::TestMockController::test_invalid_environment_variable_defaults_to_unit [32mPASSED[0m[32m [ 25%][0m
tests/test_mode_validator.py::TestMockController::test_skip_decorators [32mPASSED[0m[32m [ 29%][0m
tests/test_mode_validator.py::TestMockSafeDecorator::test_mock_safe_allows_mocks_in_unit_mode [32mPASSED[0m[32m [ 33%][0m
tests/test_mode_validator.py::TestMockSafeDecorator::test_mock_safe_allows_mocks_in_integration_mode [32mPASSED[0m[32m [ 37%][0m
tests/test_mode_validator.py::TestMockSafeDecorator::test_mock_safe_blocks_mocks_in_e2e_mode [32mPASSED[0m[32m [ 41%][0m
tests/test_mode_validator.py::TestMockSafeDecorator::test_mock_safe_with_arguments [32mPASSED[0m[32m [ 45%][0m
tests/test_mode_validator.py::TestModeIntegration::test_e2e_mode_integration [32mPASSED[0m[32m [ 50%][0m
tests/test_mode_validator.py::TestModeIntegration::test_unit_mode_integration [32mPASSED[0m[32m [ 54%][0m
tests/test_mode_validator.py::TestModeIntegration::test_mode_switching [32mPASSED[0m[32m [ 58%][0m
tests/test_mode_validator.py::TestMockControlValidation::test_mock_detection_in_unit_tests [32mPASSED[0m[32m [ 62%][0m
tests/test_mode_validator.py::TestMockControlValidation::test_mock_detection_in_e2e_tests [32mPASSED[0m[32m [ 66%][0m
tests/test_mode_validator.py::TestMockControlValidation::test_environment_consistency [32mPASSED[0m[32m [ 70%][0m
tests/test_mode_validator.py::TestMockControlValidation::test_cross_module_consistency [32mPASSED[0m[32m [ 75%][0m
tests/test_mode_validator.py::TestUnitModeSpecific::test_unit_mode_mock_usage [32mPASSED[0m[32m [ 79%][0m
tests/test_mode_validator.py::TestE2EModeSpecific::test_e2e_mode_real_components [32mPASSED[0m[32m [ 83%][0m
tests/test_mode_validator.py::TestIntegrationModeSpecific::test_integration_mode_mixed_usage [31mFAILED[0m[31m [ 87%][0m
tests/test_mode_validator.py::TestMockControllerEdgeCases::test_none_mode_handling [31mFAILED[0m[31m [ 91%][0m
tests/test_mode_validator.py::TestMockControllerEdgeCases::test_invalid_environment_handling [32mPASSED[0m[31m [ 95%][0m
tests/test_mode_validator.py::TestMockControllerEdgeCases::test_mocks_disabled_cache_consistency [32mPASSED[0m[31m [100%][0m

=================================== FAILURES ===================================
[31m[1m________ TestIntegrationModeSpecific.test_integration_mode_mixed_usage _________[0m
[1m[31mtests/test_mode_validator.py[0m:330: in test_integration_mode_mixed_usage
    assert not MockController.require_real_data()
[1m[31mE   AssertionError: assert not True[0m
[1m[31mE    +  where True = <bound method MockController.require_real_data of <class 'tests.test_modes.MockController'>>()[0m
[1m[31mE    +    where <bound method MockController.require_real_data of <class 'tests.test_modes.MockController'>> = MockController.require_real_data[0m
[31m[1m_____________ TestMockControllerEdgeCases.test_none_mode_handling ______________[0m
[1m[31mtests/test_mode_validator.py[0m:347: in test_none_mode_handling
    assert mode == TestMode.UNIT
[1m[31mE   AssertionError: assert <TestMode.E2E: 'e2e'> == <TestMode.UNIT: 'unit'>[0m
[1m[31mE    +  where <TestMode.UNIT: 'unit'> = TestMode.UNIT[0m
==================================== PASSES ====================================
[32m[1m_________________ TestMockController.test_default_mode_is_unit _________________[0m
---------------------------- Captured stdout setup -----------------------------

🧪 Test Mode: E2E
🎭 Mocks Disabled: True
🗄️  Real Database Required: True
📊 Real Data Required: True
---------------------------- Captured stderr setup -----------------------------
2025-06-15 13:50:36,349 - tests.conftest - INFO - 🧪 Test Mode: E2E
2025-06-15 13:50:36,349 - tests.conftest - INFO - 🎭 Mocks Disabled: True
2025-06-15 13:50:36,349 - tests.conftest - INFO - 🗄️  Real Database Required: True
2025-06-15 13:50:36,349 - tests.conftest - INFO - 📊 Real Data Required: True
------------------------------ Captured log setup ------------------------------
[32mINFO    [0m tests.conftest:conftest.py:65 🧪 Test Mode: E2E
[32mINFO    [0m tests.conftest:conftest.py:66 🎭 Mocks Disabled: True
[32mINFO    [0m tests.conftest:conftest.py:67 🗄️  Real Database Required: True
[32mINFO    [0m tests.conftest:conftest.py:68 📊 Real Data Required: True
[36m[1m=========================== short test summary info ============================[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_default_mode_is_unit[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_set_unit_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_set_integration_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_set_e2e_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_environment_variable_detection[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_invalid_environment_variable_defaults_to_unit[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockController::test_skip_decorators[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockSafeDecorator::test_mock_safe_allows_mocks_in_unit_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockSafeDecorator::test_mock_safe_allows_mocks_in_integration_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockSafeDecorator::test_mock_safe_blocks_mocks_in_e2e_mode[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockSafeDecorator::test_mock_safe_with_arguments[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestModeIntegration::test_e2e_mode_integration[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestModeIntegration::test_unit_mode_integration[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestModeIntegration::test_mode_switching[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControlValidation::test_mock_detection_in_unit_tests[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControlValidation::test_mock_detection_in_e2e_tests[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControlValidation::test_environment_consistency[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControlValidation::test_cross_module_consistency[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestUnitModeSpecific::test_unit_mode_mock_usage[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestE2EModeSpecific::test_e2e_mode_real_components[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControllerEdgeCases::test_invalid_environment_handling[0m
[32mPASSED[0m tests/test_mode_validator.py::[1mTestMockControllerEdgeCases::test_mocks_disabled_cache_consistency[0m
[31mFAILED[0m tests/test_mode_validator.py::[1mTestIntegrationModeSpecific::test_integration_mode_mixed_usage[0m - AssertionError: assert not True
[31mFAILED[0m tests/test_mode_validator.py::[1mTestMockControllerEdgeCases::test_none_mode_handling[0m - AssertionError: assert <TestMode.E2E: 'e2e'> == <TestMode.UNIT: 'unit'>
[31m========================= [31m[1m2 failed[0m, [32m22 passed[0m[31m in 0.05s[0m[31m =========================[0m

- ❌ **stderr**: 
- ✅ **return_code**: 1

## Integration Test Results

## Recommendations

❌ **Testing framework integration issues detected.**

Please address the errors listed above before using the testing framework.
