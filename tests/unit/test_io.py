"""Unit tests for raw IQ / WAV loading (info.md §17.1).

Covers info.md §14.1 contracts:
- complex64 native, int16 interleaved ÷32768, uint8 offset-128 ÷128
- truncate odd trailing byte, little-endian default
- WAV stereo L=I/R=Q, mono real-only
"""

from __future__ import annotations

import numpy as np
import pytest

from rf_analyzer.core.io import detect_file_type, load_iq, load_wav


def test_load_complex_iq(tmp_path):
    # Arrange
    samples = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    file_path = tmp_path / "test.iq"
    samples.tofile(file_path)

    # Act
    loaded = load_iq(str(file_path), dtype="complex64")

    # Assert
    assert len(loaded) == len(samples)
    assert np.allclose(loaded, samples, atol=1e-6)


def test_load_int16_iq(tmp_path):
    # Arrange
    raw = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
    file_path = tmp_path / "test_int16.iq"
    raw.tofile(file_path)

    # Act
    loaded = load_iq(str(file_path), dtype="int16")

    # Assert
    assert len(loaded) == 2
    assert np.isclose(loaded[0].real, 1000 / 32768.0, atol=1e-4)
    assert np.isclose(loaded[0].imag, -1000 / 32768.0, atol=1e-4)


def test_load_wav_stereo_iq(tmp_path):
    # Arrange
    import soundfile as sf

    samples = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    file_path = tmp_path / "test.wav"
    sf.write(str(file_path), samples, 100000, subtype="FLOAT")

    # Act
    iq, sample_rate = load_wav(str(file_path))

    # Assert
    assert sample_rate == 100000
    assert len(iq) == 2
    assert np.isclose(iq[0].real, 0.1, atol=1e-6)
    assert np.isclose(iq[0].imag, 0.2, atol=1e-6)


def test_load_uint8_iq(tmp_path):
    # Arrange: (raw - 128) / 128 per info.md §14.1.
    # I=192 -> 0.5, Q=64 -> -0.5; I=128 -> 0.0, Q=128 -> 0.0.
    raw = np.array([192, 64, 128, 128], dtype=np.uint8)
    file_path = tmp_path / "test_uint8.iq"
    raw.tofile(file_path)

    # Act
    loaded = load_iq(str(file_path), dtype="uint8")

    # Assert
    assert len(loaded) == 2
    assert np.isclose(loaded[0].real, 0.5, atol=1e-6)
    assert np.isclose(loaded[0].imag, -0.5, atol=1e-6)
    assert np.isclose(loaded[1].real, 0.0, atol=1e-6)
    assert np.isclose(loaded[1].imag, 0.0, atol=1e-6)


def test_load_iq_truncates_odd_trailing_byte(tmp_path):
    # Arrange: odd-length uint8 payload (3 bytes -> trailing byte dropped).
    raw_u8 = np.array([128, 128, 255], dtype=np.uint8)
    path_u8 = tmp_path / "odd_uint8.iq"
    raw_u8.tofile(path_u8)

    # Arrange: odd-count int16 payload (3 int16 -> trailing element dropped).
    raw_i16 = np.array([1000, -1000, 2000], dtype=np.int16)
    path_i16 = tmp_path / "odd_int16.iq"
    raw_i16.tofile(path_i16)

    # Act
    loaded_u8 = load_iq(str(path_u8), dtype="uint8")
    loaded_i16 = load_iq(str(path_i16), dtype="int16")

    # Assert: truncation yields exactly one complex sample each.
    assert len(loaded_u8) == 1
    assert len(loaded_i16) == 1


def test_load_int8_iq(tmp_path):
    # Arrange: HackRF signed-8 native, interleaved I/Q, ÷128.0.
    raw = np.array([64, -64, 0, 0], dtype=np.int8)
    file_path = tmp_path / "test_int8.iq"
    raw.tofile(file_path)

    # Act
    loaded = load_iq(str(file_path), dtype="int8")

    # Assert
    assert len(loaded) == 2
    assert np.isclose(loaded[0].real, 0.5, atol=1e-6)
    assert np.isclose(loaded[0].imag, -0.5, atol=1e-6)
    assert np.isclose(loaded[1].real, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("alias", "canonical", "raw_dtype", "expected_real"),
    [
        ("ci8", "int8", np.int8, 64 / 128.0),
        ("ci16", "int16", np.int16, 1024 / 32768.0),
    ],
)
def test_sigmf_dtype_aliases_match_canonical(
    tmp_path, alias, canonical, raw_dtype, expected_real
):
    """SigMF/HackRF aliases (ci8/ci16) must load identically to canonical names."""
    # Arrange
    raw = np.array([64 if raw_dtype is np.int8 else 1024, 0], dtype=raw_dtype)
    path_alias = tmp_path / f"alias_{alias}.iq"
    path_canon = tmp_path / f"canon_{canonical}.iq"
    raw.tofile(path_alias)
    raw.tofile(path_canon)

    # Act
    via_alias = load_iq(str(path_alias), dtype=alias)
    via_canon = load_iq(str(path_canon), dtype=canonical)

    # Assert
    assert np.allclose(via_alias, via_canon)
    assert np.isclose(via_alias[0].real, expected_real, atol=1e-6)


def test_load_iq_dtype_is_case_insensitive(tmp_path):
    # Arrange
    raw = np.array([64, -64], dtype=np.int8)
    file_path = tmp_path / "case.iq"
    raw.tofile(file_path)

    # Act / Assert: "CI8" and "ci8" must resolve to the same path.
    assert np.allclose(
        load_iq(str(file_path), dtype="CI8"), load_iq(str(file_path), dtype="ci8")
    )


def test_load_iq_missing_file_raises(tmp_path):
    # Arrange
    missing = tmp_path / "does_not_exist.iq"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_iq(str(missing), dtype="complex64")


def test_load_wav_missing_file_raises(tmp_path):
    # Arrange
    missing = tmp_path / "does_not_exist.wav"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        load_wav(str(missing))


def test_detect_file_type():
    # Arrange / Act / Assert
    assert detect_file_type("capture.iq") == "iq"
    assert detect_file_type("capture.wav") == "wav"
    assert detect_file_type("CAPTURE.IQ") == "iq"
    assert detect_file_type("CAPTURE.WAV") == "wav"


def test_detect_file_type_unsupported_raises():
    # Arrange / Act / Assert
    with pytest.raises(ValueError):
        detect_file_type("capture.txt")
