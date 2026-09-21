"""Milestone 1 setup check: imports + packaging resolve via pythonpath=src.

Also pins the verification harness's behaviour when the *optional* dev
requirements were not installed. That is not a hypothetical: the runtime install
(``requirements.txt``) deliberately excludes the test runner, so "ran the quick
start, then tried the documented way to prove it works" is a state a real user
lands in, and it must produce an actionable message rather than an obscure FAIL.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import rf_analyzer
import rf_analyzer.config

ROOT = Path(__file__).resolve().parents[2]


def _load_verify_mvp():
    """Load scripts/verify_mvp.py by path — scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "verify_mvp", ROOT / "scripts" / "verify_mvp.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_verdict_line_never_counts_a_skip_as_passed():
    """The printed count must not fold a skipped group into "passed".

    The real-capture group cannot run without the third-party recordings, which are
    not committed, so this line is environment-dependent. Reporting "49 passed" when
    one of the 49 never ran would be the same overstatement the project refuses to
    make about signals -- so the passed figure subtracts the skips.
    """
    verify_mvp = _load_verify_mvp()

    assert verify_mvp.verdict_line(57, 0) == "57 individual checks passed."
    line = verify_mvp.verdict_line(49, 1)
    assert "48 individual checks passed" in line
    assert "1 skipped" in line
    assert "49 individual checks passed" not in line


def test_checks_skip_is_recorded_but_not_a_failure(capsys):
    """A skip counts toward the total, is tracked separately, and is not an error."""
    verify_mvp = _load_verify_mvp()
    checks = verify_mvp.Checks()

    checks.check("a real check", True)
    checks.skip("a group that cannot run here", "because reasons")

    assert checks.total == 2
    assert checks.skipped == 1
    assert checks.ok is True, "an environment skip is not a failure"
    out = capsys.readouterr().out
    assert "[SKIP]" in out
    assert "[OK]" in out


def test_real_captures_are_skipped_not_failed_when_absent(tmp_path, capsys):
    """With no real_data/ the group must skip, leaving PASS reachable.

    This is the fresh-clone path: real_data/* is gitignored, so a clone without the
    fetch script must still be able to pass -- otherwise the documented way to prove
    the MVP works would fail for everyone who did not download 4 MB of recordings.
    """
    verify_mvp = _load_verify_mvp()
    verify_mvp.ROOT = tmp_path  # an empty repo: no real_data/SOURCES.json

    checks = verify_mvp.Checks()
    verify_mvp.check_real_captures(checks)

    assert checks.skipped == 1
    assert checks.total == 1
    assert checks.ok is True
    assert "fetch_real_data.py" in capsys.readouterr().out


def _synthetic_capture(path: Path, n: int = 4096) -> None:
    """Write a small complex64 .iq file so a check has something real to analyse."""
    import numpy as np

    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    (2.0 * bits - 1.0).astype(np.complex64).tofile(path)


def test_report_schema_check_runs_from_any_working_directory(tmp_path, monkeypatch):
    """The on-disk JSON check must not depend on where the harness was started.

    ``analyze_file`` saves its artifacts to a *relative* ``output/``, so a harness
    started outside the repo writes them there while the check looks under ROOT.
    It then silently does not run: the run still prints PASS, one check lighter,
    and nothing says why. So the check has to produce the artifact it inspects.
    """
    verify_mvp = _load_verify_mvp()

    repo = tmp_path / "repo"
    (repo / "sample_data").mkdir(parents=True)
    _synthetic_capture(repo / "sample_data" / "bpsk.iq")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # not the repo: a relative output/ lands here
    verify_mvp.ROOT = repo

    checks = verify_mvp.Checks()
    verify_mvp.check_report_schema(checks)

    assert checks.total == 5, "every schema check must run, wherever it started"
    assert checks.skipped == 0
    assert checks.ok is True


def test_a_missing_generated_capture_fails_loudly_instead_of_vanishing(
    tmp_path, capsys
):
    """A generated input that is absent must fail, not silently drop a check.

    ``tone.iq`` is produced by step [1/5] of the same run, so its absence means
    generation broke. Omitting the check instead would leave the harness printing
    PASS with a smaller total and no explanation -- the same overstatement as
    counting a skip as a pass.
    """
    verify_mvp = _load_verify_mvp()

    repo = tmp_path / "repo"
    (repo / "sample_data").mkdir(parents=True)
    _synthetic_capture(repo / "sample_data" / "fsk2.iq")  # tone.iq deliberately absent
    verify_mvp.ROOT = repo

    checks = verify_mvp.Checks()
    verify_mvp.check_2fsk_symbol_recovery(checks)

    out = capsys.readouterr().out
    tone_lines = [ln for ln in out.splitlines() if "uncorroborated 2-FSK" in ln]
    assert len(tone_lines) == 1, "the check must still be recorded, not omitted"
    assert "[FAIL]" in tone_lines[0]
    assert "tone.iq" in tone_lines[0]
    assert checks.ok is False
    assert checks.skipped == 0
    assert checks.total == 7, "the count must not shrink when an input is missing"
