"""Generate deterministic synthetic test data. See info.md §15.

Seeded (42) tone/BPSK/QPSK/2-FSK/qam16 ``.iq`` + ``.wav`` + ``*_bits.npy`` files
into ``sample_data/`` (resolved relative to the repo root so the script
works regardless of the caller's working directory).

It also emits **coded captures**: full frames built by
:mod:`rf_analyzer.core.framing` (``[sync][interleave(FEC(message))]``) with a
controlled number of channel bit errors injected. These are what prove the
error-correction and de-interleaving requirements end to end, and they ship
with a ground-truth manifest (``coded_manifest.json``) so verification can be
automated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
# The coded captures import the project's own framing module, so make src/
# importable when this script is run directly (pytest gets it from pytest.ini).
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rf_analyzer.core.framing import build_frame, modulate, modulate_2fsk

SAMPLE_RATE = 100_000
OUTPUT_DIR = REPO_ROOT / "sample_data"


def save_iq(path: Path, samples: np.ndarray):
    samples.astype(np.complex64).tofile(path)


def save_wav_iq(path: Path, samples: np.ndarray, sample_rate: float):
    stereo = np.column_stack([samples.real, samples.imag]).astype(np.float32)
    sf.write(str(path), stereo, int(sample_rate), subtype="FLOAT")


def add_awgn(samples: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(np.abs(samples) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(len(samples)) + 1j * np.random.randn(len(samples))
    )
    return samples + noise


def generate_tone(
    freq: float = 10_000, sample_rate: float = SAMPLE_RATE, duration: float = 0.1
):
    t = np.arange(int(sample_rate * duration)) / sample_rate
    samples = np.exp(1j * 2 * np.pi * freq * t)
    return samples.astype(np.complex64)


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    return 2.0 * bits.astype(np.float32) - 1.0


def generate_bpsk(
    sync_bits: np.ndarray, payload_bits: np.ndarray, snr_db: float = 30.0
):
    bits = np.concatenate([sync_bits, payload_bits])
    symbols = bits_to_bpsk(bits)
    samples = symbols.astype(np.complex64)
    samples = add_awgn(samples, snr_db)
    return bits, samples.astype(np.complex64)


def bits_to_qpsk(bits: np.ndarray) -> np.ndarray:
    if len(bits) % 2 != 0:
        bits = bits[:-1]
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    i_sym = 2.0 * i_bits.astype(np.float32) - 1.0
    q_sym = 2.0 * q_bits.astype(np.float32) - 1.0
    symbols = (i_sym + 1j * q_sym) / np.sqrt(2.0)
    return symbols.astype(np.complex64)


def generate_qpsk(
    sync_bits: np.ndarray, payload_bits: np.ndarray, snr_db: float = 30.0
):
    bits = np.concatenate([sync_bits, payload_bits])
    symbols = bits_to_qpsk(bits)
    samples = add_awgn(symbols, snr_db)
    return bits, samples.astype(np.complex64)


def generate_2fsk(
    bits: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
    symbol_duration: float = 0.001,
    freq_low: float = -5000,
    freq_high: float = 5000,
):
    """2-FSK capture. Delegates to the shared modulator (info.md §15.1 phase
    convention) so the generator and the transmit path cannot drift apart."""
    return modulate_2fsk(
        bits,
        sample_rate,
        symbol_duration=symbol_duration,
        freq_low=freq_low,
        freq_high=freq_high,
    )


def hex_to_bits(hex_string: str, bit_length: int | None = None):
    hex_string = hex_string.lower().replace("0x", "")
    value = int(hex_string, 16)
    if bit_length is None:
        bit_length = len(hex_string) * 4
    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


# ---- QAM16 helpers (SIH26147 portal bullet ii) ----


def bits_to_qam16(bits: np.ndarray) -> np.ndarray:
    """Gray-coded 16-QAM mapper, 4 bits/symbol."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = (len(bits) // 4) * 4
    if n == 0:
        return np.array([], dtype=np.complex64)
    bits = bits[:n]
    i_msb, i_lsb, q_msb, q_lsb = bits[0::4], bits[1::4], bits[2::4], bits[3::4]

    def level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        is_pos = msb == 1
        is_inner = lsb == 1
        lvl = np.where(is_inner, 1.0, 3.0)
        return np.where(is_pos, lvl, -lvl)

    i = level(i_msb, i_lsb)
    q = level(q_msb, q_lsb)
    symbols = (i + 1j * q) / np.sqrt(10.0)
    return symbols.astype(np.complex64)


def generate_qam16(
    sync_bits: np.ndarray, payload_bits: np.ndarray, snr_db: float = 30.0
):
    bits = np.concatenate([sync_bits, payload_bits])
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    symbols = bits_to_qam16(bits)
    symbols = add_awgn(symbols, snr_db)
    return bits, symbols.astype(np.complex64)


# ---- Coded captures: FEC + interleaving ground truth ----
#
# Each entry builds a real frame with rf_analyzer.core.framing and injects a
# controlled number of channel bit errors into the coded region, so the
# pipeline has to actually *correct* them rather than just parse a clean stream.
CODED_CAPTURES = [
    {
        "name": "coded_uncoded",
        "message": b"SIH26147 uncoded reference frame: CRC-16 framing only, no FEC.",
        "fec": "none",
        "interleaver": "none",
        "modulation": "BPSK",
        "errors": 0,
    },
    {
        "name": "coded_conv",
        "message": b"SIH26147 convolutional frame: K=7 r=1/2 Viterbi, Forney interleaver.",
        "fec": "conv",
        "interleaver": "convolutional",
        "modulation": "BPSK",
        "errors": 14,
    },
    {
        "name": "coded_rs",
        "message": b"SIH26147 Reed-Solomon frame: RS(255,223) over GF(256), block interleaver.",
        "fec": "rs",
        "interleaver": "block",
        "modulation": "BPSK",
        "errors": 12,
    },
    {
        "name": "coded_ldpc",
        "message": b"SIH26147 LDPC frame: (3,4)-regular min-sum belief propagation.",
        "fec": "ldpc",
        "interleaver": "pseudo-random-block",
        "modulation": "BPSK",
        "errors": 8,
    },
    {
        "name": "coded_concatenated",
        "message": b"SIH26147 concatenated frame: RS outer + convolutional inner, diagonal interleaver.",
        "fec": "concatenated",
        "interleaver": "diagonal",
        "modulation": "QPSK",
        "errors": 16,
    },
    {
        # 2-FSK is the one modulation whose samples-per-symbol is not 1, so it
        # exercises the symbol-period recovery path rather than just the
        # decision device. Without it, "demodulate FSK" would only ever be
        # proven at the bit level, never end to end through the payload chain.
        "name": "coded_fsk",
        "message": b"SIH26147 2-FSK frame: K=7 convolutional code, block interleaver, 1 ms tones.",
        "fec": "conv",
        "interleaver": "block",
        "modulation": "2-FSK",
        "errors": 10,
    },
]


def inject_bit_errors(
    bits: np.ndarray, n_errors: int, seed: int, start: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Flip ``n_errors`` random bit positions at or after ``start``.

    Returns ``(corrupted_bits, flipped_indices)``. This models a channel that
    introduces a controlled number of hard-decision errors, which is what makes
    the FEC stage's correction capability observable and reproducible.
    """
    out = np.asarray(bits, dtype=np.uint8).copy()
    if n_errors <= 0:
        return out, np.array([], dtype=np.intp)
    candidates = np.arange(start, out.size)
    if candidates.size == 0:
        return out, np.array([], dtype=np.intp)
    n = min(n_errors, candidates.size)
    rng = np.random.default_rng(seed)
    idx = rng.choice(candidates, size=n, replace=False)
    out[idx] ^= 1
    return out, np.sort(idx)


def generate_coded_captures() -> list[dict]:
    """Build every coded capture, write the ``.iq`` files and return the manifest."""
    manifest: list[dict] = []
    for i, spec in enumerate(CODED_CAPTURES):
        frame = build_frame(
            spec["message"],
            fec=spec["fec"],
            interleaver=spec["interleaver"],
        )
        sync_len = int(frame["sync_bits"].size)
        corrupted, flipped = inject_bit_errors(
            frame["bits"], spec["errors"], seed=1000 + i, start=sync_len
        )
        samples = modulate(corrupted, spec["modulation"], sample_rate=SAMPLE_RATE)
        path = OUTPUT_DIR / f"{spec['name']}.iq"
        save_iq(path, samples)

        manifest.append(
            {
                "file": path.name,
                "sample_rate": SAMPLE_RATE,
                "iq_format": "complex64",
                "sync_word": "0x1ACFFC1D",
                "fec": spec["fec"],
                "interleaver": spec["interleaver"],
                "modulation": spec["modulation"],
                "frame_bits": int(frame["frame_bits"]),
                "total_bits": int(frame["bits"].size),
                "injected_errors": int(flipped.size),
                "message_utf8": spec["message"].decode("utf-8"),
                "message_hex": spec["message"].hex(),
                "message_bytes": len(spec["message"]),
            }
        )
        print(
            f"  {path.name:24s} {spec['fec']:13s} {spec['interleaver']:20s} "
            f"{spec['modulation']:7s} frame={frame['frame_bits']:6d} bits  "
            f"errors={flipped.size}"
        )
    return manifest


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    sync_word = "0x1ACFFC1D"
    sync_bits = hex_to_bits(sync_word)
    payload_bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)

    # Tone
    tone = generate_tone()
    save_iq(OUTPUT_DIR / "tone.iq", tone)
    save_wav_iq(OUTPUT_DIR / "tone.wav", tone, SAMPLE_RATE)

    # BPSK
    bpsk_bits, bpsk_samples = generate_bpsk(sync_bits, payload_bits)
    save_iq(OUTPUT_DIR / "bpsk.iq", bpsk_samples)
    save_wav_iq(OUTPUT_DIR / "bpsk.wav", bpsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "bpsk_bits.npy", bpsk_bits)

    # QPSK
    qpsk_bits, qpsk_samples = generate_qpsk(sync_bits, payload_bits)
    save_iq(OUTPUT_DIR / "qpsk.iq", qpsk_samples)
    save_wav_iq(OUTPUT_DIR / "qpsk.wav", qpsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "qpsk_bits.npy", qpsk_bits)

    # 2-FSK
    fsk_bits = np.concatenate([sync_bits, payload_bits])
    fsk_samples = generate_2fsk(fsk_bits)
    save_iq(OUTPUT_DIR / "fsk2.iq", fsk_samples)
    save_wav_iq(OUTPUT_DIR / "fsk2.wav", fsk_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "fsk2_bits.npy", fsk_bits)

    # 16-QAM (SIH26147 portal bullet ii)
    qam_bits, qam_samples = generate_qam16(sync_bits, payload_bits)
    save_iq(OUTPUT_DIR / "qam16.iq", qam_samples)
    save_wav_iq(OUTPUT_DIR / "qam16.wav", qam_samples, SAMPLE_RATE)
    np.save(OUTPUT_DIR / "qam16_bits.npy", qam_bits)

    # Coded captures: real FEC + interleaving with injected channel errors.
    print("Generating coded captures (FEC + interleaving ground truth):")
    manifest = generate_coded_captures()
    manifest_path = OUTPUT_DIR / "coded_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/generate_test_data.py",
                "seed": 42,
                "sample_rate": SAMPLE_RATE,
                "captures": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Synthetic test data generated in sample_data/")
    print(f"Coded manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
