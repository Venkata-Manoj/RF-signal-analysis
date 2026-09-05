"""Generate deterministic synthetic test data. See info.md §15.

Seeded (42) tone/BPSK/QPSK/2-FSK ``.iq`` + ``.wav`` + ``*_bits.npy`` files
into ``sample_data/`` (resolved relative to the repo root so the script
works regardless of the caller's working directory).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 100_000
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "sample_data"


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
    samples_per_symbol = int(sample_rate * symbol_duration)
    phase = 0.0
    samples = []

    for bit in bits:
        freq = freq_high if bit == 1 else freq_low
        t = np.arange(samples_per_symbol) / sample_rate
        segment = np.exp(1j * (2 * np.pi * freq * t + phase))
        samples.append(segment)
        phase = np.angle(segment[-1])

    samples = np.concatenate(samples).astype(np.complex64)
    return samples


def hex_to_bits(hex_string: str, bit_length: int | None = None):
    hex_string = hex_string.lower().replace("0x", "")
    value = int(hex_string, 16)

    if bit_length is None:
        bit_length = len(hex_string) * 4

    bits = [(value >> (bit_length - 1 - i)) & 1 for i in range(bit_length)]
    return np.array(bits, dtype=np.uint8)


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

    print("Synthetic test data generated in sample_data/")


if __name__ == "__main__":
    main()
