# 3D DOA Estimation in High-Noise Environments

> 한국음향학회 2026 춘계학술대회 연구 프로젝트입니다. 음향 신호의 방향 추정을 위한 MUSIC, 1D 방위각 정보를 활용한 2D MUSIC, HEN 기반 희소 배열 공분산 복원 실험을 포함합니다.

Research code and selected simulation artifacts accompanying the **Korean Acoustical Society (KASA) Spring Conference 2026** paper:

**“3D High-Resolution DOA Estimation Based on Low-Rank Matrix Reconstruction in High-Noise Environments”**

The study explores microphone-array direction-of-arrival (DOA) estimation using narrowband speech, MUSIC-based spatial spectra, and high-noise acoustic simulations.

## Research overview

Three related experiments are represented in this repository:

1. **MUSIC baselines:** Comparison of 1D azimuth estimation with a uniform linear array (ULA) and 2D azimuth/elevation estimation with a planar array.
2. **Azimuth-guided planar MUSIC:** Uses estimated 1D azimuths as priors for 2D peak selection to reduce spurious-peak selection under noise. This is the main method implemented in `src/doa_az_guided_planar.py`.
3. **Sparse linear array with HEN:** Evaluates hyperbolic exponential norm (HEN) based covariance reconstruction for a sparse linear microphone array, followed by MUSIC.

**Scope note:** The azimuth-guided planar experiment and the HEN sparse-linear-array experiment are separate experiments in the paper; this repository does not claim that they form a single validated end-to-end pipeline.

## Selected experimental results

The following are **results reported in the conference paper**, not independently rerun benchmarks.

| Experiment | SNR | Azimuth RMSE | Elevation RMSE |
| --- | ---: | ---: | ---: |
| Planar array, conventional 2D MUSIC | -10 dB | 23.29° | 7.35° |
| Planar array, 1D-azimuth-guided 2D MUSIC | -10 dB | **0.67°** | **1.63°** |
| ULA, 1D MUSIC | -10 dB | 0.52° | — |
| ULA, 1D MUSIC | -25 dB | 1.17° | — |
| ULA, HEN comparison | 0 dB | 1.47° | — |
| Sparse linear array + HEN | 0 dB | **0.326°** | — |

The experiments use three simulated speech sources with azimuths `[-30°, 0°, 30°]` and elevations `[-10°, 0°, 10°]`. The planar/azimuth-guided experiments use a narrowband region around 1100 Hz; the reported HEN comparison uses 578.125 Hz. These are selected simulated conditions, not general-purpose field performance statistics.

## Repository guide

```text
src/
  doa_az_guided_planar.py      # 1D azimuth prior + planar 2D MUSIC
notebooks/
  music_snr_minus10.ipynb      # Baselines and array comparisons at -10 dB
  music_snr_minus25.ipynb      # Includes the -25 dB ULA test
  linear_music_and_hen.ipynb   # ULA baseline and sparse-array HEN experiment
samples/
  male26.flac
  female19.flac
  female8975.flac             # Three main sample utterances
  additional/                # Additional exploratory audio
  SPEAKERS.TXT               # Source-corpus speaker metadata
paper/
  2026_KASA_Spring_3D_DOA.pdf # Conference paper
archive/
  initial_array_comparison.ipynb
  early_linear_music.py
  ocean_simulation/          # Separate acoustic simulation assets
  editor_settings/           # Previously tracked editor configuration
docs/
  experiment_mapping.md      # Paper-to-code mapping and known limitations
```

The `archive/` directory retains exploratory material for reference. The ocean simulation configuration files are **not** part of the reported indoor DOA experiments.

## Getting started

The code is research-oriented and has **not** been independently rerun or packaged as a one-command reproducibility release.

```bash
python -m pip install -r requirements.txt
```

Before running `src/doa_az_guided_planar.py`, edit the `LIBRISPEECH_FILES` paths at the top of the file. The original project used Windows absolute paths; when running from the repository root, the three primary files can instead be referenced as:

```python
LIBRISPEECH_FILES = [
    "samples/male26.flac",
    "samples/female19.flac",
    "samples/female8975.flac",
]
```

The notebooks also contain experiment-specific settings and may depend on their execution order. Saved notebook outputs are provided as a record of prior experiments.

## Data and attribution

The audio samples were selected from the **LibriSpeech** speech corpus. See [LibriSpeech on OpenSLR](https://www.openslr.org/12/) for dataset information and applicable redistribution/attribution terms. The repository's speaker metadata comes from the source corpus. The linked paper explains the experimental setup and cites the underlying HEN method.

## Paper

[Read the KASA Spring 2026 conference paper](paper/2026_KASA_Spring_3D_DOA.pdf)

This repository is intended to document the research methods and code organization. It is not a production audio localization system.
