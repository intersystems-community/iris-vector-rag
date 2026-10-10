"""The pure note-search modules can be copied out of this package and still import.

Baystate vendors them into an ISC-internal repo and runs them in IRIS's embedded Python,
where torch, the network and the rest of iris_vector_rag are absent. This test copies the
pure modules into a scratch package under another name, forbids every import that is not
standard library, and runs the entry points.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import iris_vector_rag.note_search as note_search

PURE = [
    "__init__",
    "tokens",
    "phrase",
    "sections",
    "polarity",
    "rollup",
    "api",
    "concepts",
]

SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
allowed = set(sys.stdlib_module_names) | {"vendored_ns"}

class Forbid:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in allowed:
            raise ImportError("forbidden import: " + name)

sys.meta_path.insert(0, Forbid())
from vendored_ns import api, concepts, phrase, polarity, rollup, sections, tokens
assert concepts.DictionaryTagger({"C-X": ["chest pain"]})("Chest pain.")[0].start == 0
result = api.judge_section("Denies chest pain.", "chest pain")
assert result["polarity"] == "negated", result
assert sections.split_sections("HPI: x", sections.default_heading_detector)[0].label == "hpi"
print("ok")
"""


@pytest.fixture
def vendored(tmp_path):
    package = tmp_path / "vendored_ns"
    package.mkdir()
    source = Path(note_search.__file__).parent
    for name in PURE:
        shutil.copy(source / f"{name}.py", package / f"{name}.py")
    return tmp_path


def test_the_pure_modules_run_with_only_the_standard_library(vendored):
    done = subprocess.run(
        [sys.executable, "-I", "-c", SCRIPT, str(vendored)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_no_pure_module_imports_the_parent_package_by_absolute_name():
    source = Path(note_search.__file__).parent
    for name in PURE:
        text = (source / f"{name}.py").read_text()
        assert "from iris_vector_rag" not in text, name
        assert "import iris_vector_rag" not in text, name
