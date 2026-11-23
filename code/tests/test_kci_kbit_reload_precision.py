"""Isolate HF/PEFT imports from the suite's pinned-tokenizer provenance tests."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("case", [
    "test_saved_qlora_preserves_action_probabilities_on_reload",
    "test_frozen_base_matches_adapter_disabled_reference",
])
def test_precision_in_isolated_process(case, tmp_path):
    if importlib.util.find_spec("peft") is None:
        pytest.skip("Optional HF/PEFT integration dependencies unavailable")
    helper = Path(__file__).with_name("kbit_precision_cases.py")
    result = subprocess.run([
        sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        str(helper) + "::" + case, "--basetemp", str(tmp_path / "child"),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace",
       env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
