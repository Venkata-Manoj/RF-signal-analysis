"""CLI --receiver flag reaches the pipeline report (scripts/run_pipeline.py).

Direct ``main()`` runs on a tiny synthetic BPSK ``.iq`` in ``tmp_path``
(decode off, so each run is well under a second):

* ``--receiver naive`` is reported as
  ``demodulation.receiver.requested == "naive"`` with ``path == "naive"``.
* omitting the flag (default) and passing ``--receiver auto`` both report
  ``requested == "auto"``.
* ``--help`` mentions the flag; an invalid value exits nonzero via
  ``argparse`` choices (no third mode exists).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rf_analyzer.core.framing import build_frame, modulate

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"

SYNC_WORD = "0x1ACFFC1D"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_pipeline_cli", RUN_PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline_cli"] = module
    spec.loader.exec_module(module)
    return module


def _make_iq(path: Path) -> Path:
    """Tiny framed BPSK burst (1 sample/symbol), decode-independent."""
    frame = build_frame(b"CLI recv", fec="none", interleaver="none")
    samples = modulate(frame["bits"], "BPSK")
    np.asarray(samples, dtype=np.complex64).tofile(path)
    return path


def _run(cli, args: list[str]) -> int:
    return int(cli.main(args))


def _report(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


def _base_args(iq: Path, out: Path) -> list[str]:
    return [
        str(iq),
        "--sample-rate",
        "100000",
        "--iq-format",
        "complex64",
        "--modulation",
        "BPSK",
        "--sync-word",
        SYNC_WORD,
        "--no-decode",
        "--out",
        str(out),
    ]


def test_cli_receiver_naive_reaches_report(tmp_path):
    cli = _load_cli()
    iq = _make_iq(tmp_path / "cli_naive.iq")
    out = tmp_path / "naive.json"
    rc = _run(cli, [*_base_args(iq, out), "--receiver", "naive"])
    assert rc == 0
    receiver = _report(out)["demodulation"]["receiver"]
    assert receiver["requested"] == "naive"
    assert receiver["path"] == "naive"


def test_cli_receiver_default_is_auto(tmp_path):
    cli = _load_cli()
    iq = _make_iq(tmp_path / "cli_default.iq")
    out = tmp_path / "default.json"
    rc = _run(cli, _base_args(iq, out))  # no --receiver flag
    assert rc == 0
    receiver = _report(out)["demodulation"]["receiver"]
    assert receiver["requested"] == "auto"


def test_cli_receiver_explicit_auto(tmp_path):
    cli = _load_cli()
    iq = _make_iq(tmp_path / "cli_auto.iq")
    out = tmp_path / "auto.json"
    rc = _run(cli, [*_base_args(iq, out), "--receiver", "auto"])
    assert rc == 0
    receiver = _report(out)["demodulation"]["receiver"]
    assert receiver["requested"] == "auto"


def test_cli_help_mentions_receiver_flag():
    cli = _load_cli()
    help_text = cli.build_parser().format_help()
    assert "--receiver" in help_text
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_cli_invalid_receiver_exits_nonzero(tmp_path):
    cli = _load_cli()
    iq = _make_iq(tmp_path / "cli_bogus.iq")
    out = tmp_path / "bogus.json"
    with pytest.raises(SystemExit) as exc:
        cli.main([*_base_args(iq, out), "--receiver", "bogus"])
    assert int(exc.value.code) != 0
