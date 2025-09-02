import lit.formats
import psutil
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Assignment2'

# testFormat: The test format to use to interpret tests.
config.test_format = lit.formats.ShTest(execute_external=False)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.ll']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root
config.target_triple = '(unused)'

# Set timeout
supported, errormsg = lit_config.maxIndividualTestTimeIsSupported
if supported:
    config.available_features.add("lit-max-individual-test-time")
    lit_config.maxIndividualTestTime = 5
else:
    lit_config.warning(
        "Setting a timeout per test not supported. "
        + errormsg
        + " Some tests will be skipped."
    )

