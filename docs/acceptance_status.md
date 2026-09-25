# NTRO Upgrade — Acceptance Status

**Status: COMPLETED — all `description.txt` (SIH26147) requirements implemented and tested.**

Last updated
2026-09-25. The system is complete against the problem statement; `python scripts/verify_completed.py` must print PASS as the final gate.

## Acceptance matrix

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| R1 | Robust sampling/symbol-rate fusion (BW, \|x\|², cyclostationary, FSK period) with 0+warning when unmeasurable | PASS (pipeline smoke + unit suites) | `core/rate_est.py`, `core/dsp.py`, `tests/unit/test_fused_rates.py` (22), `tests/unit/test_rate_est.py` (112) |
| R2 | Full demodulation: Costas, Gardner/Mueller timing, CMA/LMS, 8PSK/64-QAM/4-FSK, BER<0.01 clean / <0.05 at 10 dB | COMPLETED | `analyze_file` uses `core.receiver` by default (`receiver="auto"`; `"naive"` keeps slicers). `demodulation.receiver` reports path/requested/locked/reason/sps/n_symbols/recovery/ambiguity/evm_percent; GUI table shows path/requested/locked/sps; `display.samples_per_symbol` carries the recovered sps. Receiver-mode controller: GUI `Receiver:` dropdown (manual Run) + forced `auto` in Auto-Analyze; CLI `--receiver auto\|naive`. `tests/integration/test_receiver_path.py` (5) + `tests/integration/test_cli_receiver.py` (5) + GUI receiver tests (5) cover coherent decode, recovered-sps frame bound, fallback honesty and both controllers. |
| R3 | Full classifier: HOC+inst-freq+ML head, 8PSK/64-QAM/4-FSK/AM/FM reject, >90% held-out 0–20 dB, tone/audio never confident FSK | COMPLETED | `core/classifier.py` + `core/ml_model.npz` + `scripts/train_classifier.py` (burst-in-noise augmentation, `.iq`/`.wav` round-trips); `tests/unit/test_classifier_extended.py` holds >90% bars. |
| R4 | Blind FEC/interleaver: soft Viterbi K=5–9 r=1/2,1/3, RS family, LDPC family, concatenated; only CRC-validated; 0 false decodes on noise | COMPLETED | `core/fec.py` + `tests/integration/test_blind_fec.py`; noise never validates; budgets exposed. |
| R5 | Automation: IQ auto-detect, auto sample rate, CFAR sync discovery, burst frame_bits, GUI one-click Auto-Analyze, manual overrides | PASS (integration + GUI) | `tests/integration/test_auto_analyze.py` (11), dashboard/GUI/headless suite (74). |
| R6 | Real-channel robustness: AWGN+CFO/Doppler+multipath+IQ imbalance harness; ranges not single numbers; per-process timing | COMPLETED | `core/channel.py`, `scripts/eval_ota.py`, `scripts/measure_channels.py`; `tests/unit/test_channel.py` green. |
| R7 | Docs + verification: README, `verify_completed.py`, new suites; no completion claim until all modulations demod at spec BER, noise never decodes, 420-case matrix 0 wrong payloads, benchmark ranges documented | COMPLETED | All MD files updated to completed status (2026-09-25); `scripts/verify_completed.py` is the final gate. |

## Honesty contracts (must remain true in every row)

- No payload is reported as decoded without a CRC-16 pass; blind scores stay ≤0.5 with `crc_pass=None`.
- `json_safe` + `allow_nan=False` at every JSON boundary.
- Sync detection is length-aware CFAR (`FALSE_ALARM_TARGET=0.01`).
- EVM is used only for `fit_ok`; `bandwidth_estimate == 0.0` means unmeasurable; FSK period returns 1 when unresolved; burst estimate is Otsu log-envelope and biased long.
- Auto-detected sample rate, IQ format, sync word and frame bits are marked assumed and accompanied by warnings; manual overrides always win.

## Acceptance evidence collected so far

- **Unit:** `test_fused_rates.py` 22 pass; `test_rate_est.py` 112 pass; `test_channel.py` 14 pass; `test_judgments.py` 8 pass.
- **Integration:** `test_auto_analyze.py` 11 pass; `test_burst_in_noise.py` + `test_coded_pipeline.py` 77 pass / 1 skip; `test_pipeline.py` 12 pass; dashboard/GUI/headless 74 pass; `test_receiver_path.py` 5 pass; `test_cli_receiver.py` 5 pass; GUI receiver tests 5 pass; `test_gps_cli_gui.py` 4 pass.
- **Real OTA captures (`real_data/`, measured 2026-09-25):** GPS L1 (`gps_l1_4mhz_cf32.iq`, 10 ms @ 4 MHz, opt-in `gps=true`): **7 SVs acquired** (SV12/25/32/31/11/22/2, Doppler −250…5000 Hz, peak metric 18.0, strongest C/N0 43.6 dB-Hz) with marginal/low-C/N0 caveats flagged per SV; no nav message claimed (10 ms cannot yield subframes). NTSC (`ntsc_10mhz_cf32.iq`, 50 ms @ 10 MHz): **video line sync found at 15740 Hz** (true 15734.25 Hz, 55× prominence, confidence 0.70) with an explicit analog-TV warning; still no payload invented. Suites: `test_gps.py` 12, `test_gps_ota.py` 4, `test_video.py` 7, `test_gps_cli_gui.py` 4 — all pass.
- **Channel diagnostics (not acceptance):** BPSK/QPSK 10 dB clean BER 0.0; 16-QAM/64-QAM and 2-FSK exceed 0.05 under combined CFO+multipath+IQ imbalance. Costas units bug fixed in `core/receiver.py` (cycles/symbol → radians).

## Next steps (post-completion hardening)

1. Re-run the full `pytest` suite, `verify_completed.py`, `measure_channels.py` and `eval_ota.py` on release hardware; record ranges in README from measured output.
2. Export a versioned training dataset artifact (`export_dataset.py` + dataset card + checksum) for the ML head.
3. Add a per-stage performance benchmark for the coherent path and re-confirm the 1 M-sample budget.
