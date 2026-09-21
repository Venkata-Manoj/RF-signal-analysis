"""Milestone 1 setup check: imports + packaging resolve via pythonpath=src.

Also pins the verification harness's behaviour when the *optional* dev
requirements were not installed. That is not a hypothetical: the runtime install
(``requirements.txt``) deliberately excludes the test runner, so "ran the quick
start, then tried the documented way to prove it works" is a state a real user
lands in, and it must produce an actionable message rather than an obscure FAIL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import rf_analyzer
import rf_analyzer.config

ROOT = Path(__file__).resolve().parents[2]

#: Blocks ``import pytest`` in the child process, so the runtime-only install can
#: be reproduced without building a second virtualenv.
_BLOCK_PYTEST = """\
import builtins

_real_import = builtins.__import__


def _blocked(name, *args, **kwargs):
    if name == "pytest":
        raise ImportError("blocked for test")
    return _real_import(name, *args, **kwargs)


builtins.__import__ = _blocked
"""


def test_package_imports():
    assert rf_analyzer.__version__ == "0.1.0"
    assert rf_analyzer.config.TOOL_NAME.startswith("RF Signal")


def test_verify_mvp_explains_a_missing_dev_install(tmp_path):
    """A missing test runner must name the fix and exit 2, not fail obscurely.

    The exit code matters as much as the message: ``1`` means a check genuinely
    failed, so reusing it here would make "you did not install the dev
    requirements" indistinguishable from "the project is broken" to any script
    that runs this.
    """
    # `sitecustomize` is auto-imported at interpreter start when it is on the
    # path, so this blocks pytest in the child before verify_mvp does anything.
    (tmp_path / "sitecustomize.py").write_text(_BLOCK_PYTEST, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_mvp.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )

    assert proc.returncode == 2, (
        "a missing dev install must exit 2, distinct from the exit 1 a failed "
        f"check uses; got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pytest is not installed" in proc.stdout
    assert (
        "pip install -r requirements-dev.txt" in proc.stdout
    ), "the message must be copy-pasteable, not just descriptive"
    assert (
        "requirements.txt alone is enough" in proc.stdout
    ), "it must not read as though the runtime install was wrong"
