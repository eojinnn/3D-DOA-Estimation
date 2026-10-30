# Experiment-to-code mapping

This guide identifies how the uploaded files correspond to the experiments described in the KASA Spring 2026 paper.

| Paper experiment / figure | Implementation or saved output | Notes |
| --- | --- | --- |
| Narrowband speech signal and microphone simulation (Fig. 1) | `src/doa_az_guided_planar.py`; selected baseline notebooks | STFT and frequency-band filtering; original scripts use local paths |
| 1D ULA MUSIC, -10 dB (Fig. 5) | `notebooks/music_snr_minus10.ipynb` | Saved ULA output: approximately 0.5204° azimuth RMSE |
| 1D ULA MUSIC, -25 dB (Fig. 6) | `notebooks/music_snr_minus25.ipynb` | Saved ULA output: approximately 1.1726° azimuth RMSE |
| Conventional planar 2D MUSIC, -10 dB (Fig. 4) | `notebooks/music_snr_minus25.ipynb` | The notebook changes the linear ULA SNR to -25 dB but retains a -10 dB planar test; saved planar output: approximately 23.2863° azimuth / 7.3485° elevation RMSE |
| Azimuth-guided planar MUSIC (discussion, p. 3) | `src/doa_az_guided_planar.py` | Implementation of azimuth priors and 2D peak selection; rerun required to confirm the paper's 0.67° / 1.63° values |
| Sparse ULA + HEN reconstruction (Figs. 7–9) | `notebooks/linear_music_and_hen.ipynb` | Saved output includes HEN RMSE approximately 0.3266° and ULA baseline approximately 1.4731° |

## Notes on different files

- `music_snr_minus10.ipynb` and `music_snr_minus25.ipynb` have almost the same implementation but **different SNR conditions and saved outputs**. Both are kept intentionally.
- `linear_music_and_hen.ipynb` contains an executable HEN implementation and a ULA comparison in different cells. Some functions are redefined between cells, so execution order matters.
- `src/doa_az_guided_planar.py` is the most directly relevant implementation of azimuth-guided 2D MUSIC.
- `archive/initial_array_comparison.ipynb` compares earlier array designs, including a tetrahedral-array experiment, and is not the final method.
- `archive/early_linear_music.py` includes an exploratory multi-source linear array setup and does not directly correspond to the final three-source experimental configuration.
- `archive/ocean_simulation/` contains unrelated acoustic propagation configuration assets retained from the original upload.

## Verification status

File assignments were made by inspecting source definitions, notebook outputs, and matching selected reported metrics. The original script contents and saved notebook outputs were retained **without altering algorithm logic**. No full rerun, dependency pinning, or figure-by-figure reproducibility verification has been performed.
