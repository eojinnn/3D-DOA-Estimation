# ============================================================
# Global config
# ============================================================

import math
import random
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import stft, resample_poly, find_peaks
from scipy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import maximum_filter
import pyroomacoustics as pra

np.random.seed(42)
random.seed(42)
# ------------------------------------------------------------
# Acoustic / signal basics
# ------------------------------------------------------------
FS = 16000                 # sampling rate
C = 343.0                  # speed of sound [m/s]
DURATION_SEC = 3.0         # use same duration for all utterances


# ------------------------------------------------------------
# Input speech files
# ------------------------------------------------------------
LIBRISPEECH_FILES = [
    "C:/26'음향학회 춘계학술대회/male26.flac",
    "C:/26'음향학회 춘계학술대회/male27.flac",
    "C:/26'음향학회 춘계학술대회/female19.flac",
    "C:/26'음향학회 춘계학술대회/female32.flac",
    "C:/26'음향학회 춘계학술대회/female8975.flac",

]


# ------------------------------------------------------------
# Source configuration
# Standard spherical convention:
#   azimuth   : xy-plane from +x toward +y
#   elevation : from xy-plane toward +z
# ------------------------------------------------------------
N_SOURCES = 5

SOURCE_AZIMUTH_DEG = np.array([-30.0, -10.0,0.0,10.0,30.0], dtype=np.float64) #-35, 5, 40
SOURCE_ELEVATION_DEG = np.array([0.0,0.0,0.0,0.0,0.0], dtype=np.float64)
SOURCE_DISTANCES_M = np.array([3.4,3.2,3.0, 2.8,2.6], dtype=np.float64) #2.5, 2.8, 3.2

ARRAY_CENTER = np.array([0.0, 0.0, 1.5], dtype=np.float64)


# ------------------------------------------------------------
# Array comparison setting
#
# same_spacing:
#   linear adjacent spacing = tetra edge length = BASE_VALUE_M
#
# same_baseline:
#   linear max baseline = tetra max baseline = BASE_VALUE_M
#   -> linear spacing = BASE_VALUE_M / 3
#   -> tetra edge     = BASE_VALUE_M
# ------------------------------------------------------------
COMPARE_MODE = "same_spacing"   # "same_spacing" or "same_baseline"
BASE_VALUE_M = 0.29              # 8 cm


# ------------------------------------------------------------
# Frequency selection
# Recommended: use fixed frequency first for stable comparison
# ------------------------------------------------------------
USE_FIXED_CENTER_HZ = False
FIXED_CENTER_HZ = 1800.0

SEARCH_FMIN_HZ = 300.0
SEARCH_FMAX_HZ = 3400.0
BANDWIDTH_HZ = 150.0


# ------------------------------------------------------------
# STFT
# ------------------------------------------------------------
STFT_NFFT = 1024
STFT_HOP = 256


# ------------------------------------------------------------
# MUSIC scan grids
# Linear: azimuth-only 1D scan
# Tetra : azimuth + elevation 2D scan
# ------------------------------------------------------------
AZ_GRID_DEG = np.linspace(-90.0, 90.0, 1801)     # 0.1 deg resolution
EL_GRID_DEG = np.linspace(-60.0, 60.0, 121)      # 1 deg resolution


# ------------------------------------------------------------
# Optional plotting / debug
# ------------------------------------------------------------
PLOT_RAW_AND_BAND_SPECTRA = True
PLOT_MIC_SIGNALS = True
PLOT_LINEAR_MUSIC = True
PLOT_TETRA_MUSIC = True


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
assert len(LIBRISPEECH_FILES) == N_SOURCES, \
    f"Number of files ({len(LIBRISPEECH_FILES)}) must match N_SOURCES ({N_SOURCES})"

assert len(SOURCE_AZIMUTH_DEG) == N_SOURCES
assert len(SOURCE_ELEVATION_DEG) == N_SOURCES
assert len(SOURCE_DISTANCES_M) == N_SOURCES

print("Global config loaded.")
print(f"N_SOURCES           : {N_SOURCES}")
print(f"COMPARE_MODE        : {COMPARE_MODE}")
print(f"BASE_VALUE_M        : {BASE_VALUE_M:.4f} m")
print(f"USE_FIXED_CENTER_HZ : {USE_FIXED_CENTER_HZ}")
if USE_FIXED_CENTER_HZ:
    print(f"FIXED_CENTER_HZ     : {FIXED_CENTER_HZ:.1f} Hz")

# ============================================================
# Common helpers
# ============================================================

# ------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------
def read_audio_mono_resample(path: str, target_fs: int) -> np.ndarray:
    x, fs = sf.read(path)

    if x.ndim > 1:
        x = np.mean(x, axis=1)

    x = x.astype(np.float64)

    if fs != target_fs:
        print('sampling rate unmatched resampling...')
        g = math.gcd(fs, target_fs)
        up = target_fs // g
        down = fs // g
        x = resample_poly(x, up, down)

    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-12)
    return x


def center_crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) >= target_len:
        start = (len(x) - target_len) // 2
        return x[start:start + target_len].copy()

    print("audio shorter than target, zero padding...")
    y = np.zeros(target_len, dtype=np.float64)
    y[:len(x)] = x
    return y


def average_speech_spectrum(x: np.ndarray, fs: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    f, _, Z = stft(x, fs=fs, window="hann", nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)

    mag = np.abs(Z)
    frame_energy = np.mean(mag**2, axis=0)

    thr = np.percentile(frame_energy, 10.0)
    valid = frame_energy > thr
    if not np.any(valid):
        print('error: no energy bigger than threshold')
        valid = np.ones_like(frame_energy, dtype=bool)

    avg_mag = np.mean(mag[:, valid], axis=1)
    return f, avg_mag


def choose_common_dominant_band(signals: list[np.ndarray], fs: int, fmin_hz: float, fmax_hz: float,) -> float:
    stacked = None
    freqs = None

    for x in signals:
        f, avg_mag = average_speech_spectrum(x, fs=fs)
        avg_mag = avg_mag / (np.max(avg_mag) + 1e-12)

        if stacked is None:
            stacked = avg_mag
            freqs = f
        else:
            stacked += avg_mag

    mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(mask):
        raise ValueError("No FFT bins inside search range.")

    f_sel = freqs[mask]
    a_sel = stacked[mask]
    center_hz = f_sel[np.argmax(a_sel)]
    return float(center_hz)


def get_center_frequency(signals: list[np.ndarray]) -> float:
    if USE_FIXED_CENTER_HZ:
        return float(FIXED_CENTER_HZ)

    return choose_common_dominant_band(
        signals=signals,
        fs=FS,
        fmin_hz=SEARCH_FMIN_HZ,
        fmax_hz=SEARCH_FMAX_HZ,
    )


def fft_keep_only_band(x: np.ndarray, fs: int, center_hz: float, bandwidth_hz: float) -> np.ndarray:
    n = len(x)
    X = rfft(x)
    f = rfftfreq(n, d=1.0 / fs)

    half_bw = bandwidth_hz / 2.0
    mask = (f >= center_hz - half_bw) & (f <= center_hz + half_bw)

    Y = np.zeros_like(X)
    Y[mask] = X[mask]
    y = irfft(Y, n=n)

    y = y - np.mean(y)
    y = y / (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float64)


def prepare_narrowband_sources(file_paths: list[str], duration_sec: float) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    target_len = int(FS * duration_sec)

    raw_sources = []
    for path in file_paths:
        x = read_audio_mono_resample(path, FS)
        x = center_crop_or_pad(x, target_len)
        raw_sources.append(x)

    center_hz = get_center_frequency(raw_sources)

    narrow_sources = [
        fft_keep_only_band(
            x,
            fs=FS,
            center_hz=center_hz,
            bandwidth_hz=BANDWIDTH_HZ,
        )
        for x in raw_sources
    ]

    return raw_sources, narrow_sources, center_hz


# ------------------------------------------------------------
# Common geometry / source helpers
# ------------------------------------------------------------
def sph_to_cart(
    center_xyz: np.ndarray,
    az_deg: float,
    el_deg: float,
    r: float,
) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    x = center_xyz[0] + r * np.cos(el) * np.cos(az)
    y = center_xyz[1] + r * np.cos(el) * np.sin(az)
    z = center_xyz[2] + r * np.sin(el)

    return np.array([x, y, z], dtype=np.float64)


def build_source_positions() -> list[np.ndarray]:
    positions = []
    for az_deg, el_deg, dist_m in zip(
        SOURCE_AZIMUTH_DEG,
        SOURCE_ELEVATION_DEG,
        SOURCE_DISTANCES_M,
    ):
        positions.append(
            sph_to_cart(
                center_xyz=ARRAY_CENTER,
                az_deg=az_deg,
                el_deg=el_deg,
                r=dist_m,
            )
        )
    return positions

def true_linear_broadside_deg(
    source_positions: list[np.ndarray],
    array_center: np.ndarray,
) -> np.ndarray:
    """
    x-axis ULA 기준 true broadside angle 계산
    0 deg  : +y (broadside)
    - angle: +x 쪽으로 기울어짐
    + angle: -x 쪽으로 기울어짐
    """
    angles = []
    for p in source_positions:
        rel = np.asarray(p, dtype=np.float64) - np.asarray(array_center, dtype=np.float64)
        dx, dy = rel[0], rel[1]
        theta_deg = np.rad2deg(np.arctan2(-dx, dy))
        angles.append(theta_deg)
    return np.array(angles, dtype=np.float64)


# ------------------------------------------------------------
# Common STFT / covariance helpers
# ------------------------------------------------------------
def multichannel_stft(
    y_mics: np.ndarray,
    fs: int,
    n_fft: int,
    hop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z_all = []
    f_ref = None
    t_ref = None

    for m in range(y_mics.shape[0]):
        f, t, Z = stft(
            y_mics[m],  
            fs=fs,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary=None,
            padded=False,
        )
        Z_all.append(Z)

        if f_ref is None:
            f_ref = f
            t_ref = t

    Z_all = np.stack(Z_all, axis=0)  # M x F x T
    return f_ref, t_ref, Z_all


def extract_frequency_bin_snapshots(
    y_mics: np.ndarray,
    fs: int,
    target_hz: float,
    n_fft: int,
    hop: int,
) -> tuple[float, np.ndarray]:
    f, _, Z = multichannel_stft(y_mics, fs, n_fft=n_fft, hop=hop)
    k = int(np.argmin(np.abs(f - target_hz)))
    print('shape after stft', Z.shape)
    print('target center_hz', target_hz)
    print('calculated center_hz', f[k])
    X_bin = Z[:, k, :]  # M x N_snapshots
    return float(f[k]), X_bin


def sample_covariance(X: np.ndarray) -> np.ndarray:
    return (X @ X.conj().T) / X.shape[1]


# ------------------------------------------------------------
# Common simulation / metric helpers
# ------------------------------------------------------------
def simulate_array_signals(
    mic_locs: np.ndarray,
    source_positions: list[np.ndarray],
    source_signals: list[np.ndarray],
    fs: int,
) -> np.ndarray:
    room = pra.AnechoicRoom(fs=fs, dim=3)
    room.add_microphone_array(mic_locs)

    for pos, sig in zip(source_positions, source_signals):
        room.add_source(pos, signal=sig)
    snr_db=0
    room.simulate(snr=snr_db)
    return room.mic_array.signals


def azimuth_rmse_deg(
    true_az_deg: np.ndarray,
    est_az_deg: np.ndarray,
) -> float:
    true_az = np.sort(np.asarray(true_az_deg, dtype=np.float64))
    est_az = np.sort(np.asarray(est_az_deg, dtype=np.float64))

    if len(true_az) != len(est_az):
        raise ValueError(
            f"RMSE source count mismatch: true={len(true_az)}, est={len(est_az)}"
        )

    return float(np.sqrt(np.mean((est_az - true_az) ** 2)))


# ============================================================
# Linear array helpers
# ============================================================

# ------------------------------------------------------------
# Linear array helpers
# ------------------------------------------------------------
def get_linear_spacing_from_config() -> float:
    """
    same_spacing:
        linear adjacent spacing = BASE_VALUE_M

    same_baseline:
        linear max baseline = BASE_VALUE_M
        -> adjacent spacing = BASE_VALUE_M / 3
    """
    if COMPARE_MODE == "same_spacing":
        return float(BASE_VALUE_M)

    if COMPARE_MODE == "same_baseline":
        return float(BASE_VALUE_M / 3.0)

    raise ValueError("COMPARE_MODE must be 'same_spacing' or 'same_baseline'")


def make_linear_ula(
    center_xyz: np.ndarray,
    # spacing_m: float,
) -> np.ndarray:
    """
    4-mic uniform linear array on y-axis.
    Returns shape (3, 4)
    """
    # offsets = np.array([-1.5,-0.5,0.5,1.5], dtype=np.float64) * spacing_m
    offsets = np.array([-0.435, -0.145, 0.145, 0.435], dtype=np.float64)

    x = np.full(4, center_xyz[0], dtype=np.float64)
    y = center_xyz[1] + offsets
    z = np.full(4, center_xyz[2], dtype=np.float64)

    return np.vstack([x, y, z])


def build_linear_array() -> tuple[np.ndarray, float]:
    spacing_m = get_linear_spacing_from_config()
    mic_locs = make_linear_ula(
        center_xyz=ARRAY_CENTER,
        # spacing_m=spacing_m,
    )
    return mic_locs, spacing_m

# ------------------------------------------------------------
# Linear azimuth-only MUSIC
# ------------------------------------------------------------

def steering_vector_linear_azimuth(
    mic_positions_xyz: np.ndarray,
    array_center_xyz: np.ndarray,
    freq_hz: float,
    az_deg: float,
    c: float = 343.0,
) -> np.ndarray:
    """
    Linear ULA azimuth-only steering vector.
    Elevation is assumed to be 0 deg.
    """
    az = np.deg2rad(az_deg)
    y_rel = mic_positions_xyz[1, :] - array_center_xyz[1]
    phase = 2.0j * np.pi * freq_hz / c * y_rel * np.sin(az)
    return np.exp(phase)[:, None]


def music_spectrum_linear_1d(
    R: np.ndarray,
    mic_positions_xyz: np.ndarray,
    array_center_xyz: np.ndarray,
    freq_hz: float,
    n_sources: int,
    az_grid_deg: np.ndarray,
    c: float = 343.0,
) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    En = eigvecs[:, n_sources:]
    P = np.zeros(len(az_grid_deg), dtype=np.float64)

    for i, az in enumerate(az_grid_deg):
        a = steering_vector_linear_azimuth(
            mic_positions_xyz=mic_positions_xyz,
            array_center_xyz=array_center_xyz,
            freq_hz=freq_hz,
            az_deg=az,
            c=c,
        )
        denom = (a.conj().T @ En @ En.conj().T @ a).squeeze()
        P[i] = 1.0 / (np.abs(denom) + 1e-12)

    P = P / (np.max(P) + 1e-12)
    return P

def estimate_music_peaks_1d(
    P: np.ndarray,
    az_grid_deg: np.ndarray,
    n_sources: int,
    min_peak_distance_deg: float = 5.0,
) -> list[tuple[float, float]]:
    step_deg = float(np.abs(az_grid_deg[1] - az_grid_deg[0]))
    min_distance_bins = max(1, int(min_peak_distance_deg / step_deg))

    peaks, _ = find_peaks(P, distance=min_distance_bins)
    if len(peaks) == 0:
        return []

    peak_vals = P[peaks]
    order = np.argsort(peak_vals)[::-1]
    chosen = peaks[order[:n_sources]]

    results = [(float(az_grid_deg[i]), float(P[i])) for i in chosen]
    results.sort(key=lambda x: x[0])
    return results

def run_linear_music_pipeline(
    mic_locs: np.ndarray,
    source_positions: list[np.ndarray],
    source_signals: list[np.ndarray],
    center_hz: float,
) -> dict:
    y_mics = simulate_array_signals(
        mic_locs=mic_locs,
        source_positions=source_positions,
        source_signals=source_signals,
        fs=FS,
    )

    chosen_bin_hz, X_bin = extract_frequency_bin_snapshots(
        y_mics=y_mics,
        fs=FS,
        target_hz=center_hz,
        n_fft=STFT_NFFT,
        hop=STFT_HOP,
    )
    print('X_bin shape:', X_bin.shape)

    R = sample_covariance(X_bin)
    print('covariance shape:', R.shape)

    P = music_spectrum_linear_1d(
        R=R,
        mic_positions_xyz=mic_locs,
        array_center_xyz=ARRAY_CENTER,
        freq_hz=chosen_bin_hz,
        n_sources=N_SOURCES,
        az_grid_deg=AZ_GRID_DEG,
        c=C,
    )
    
    peaks = estimate_music_peaks_1d(
        P=P,
        az_grid_deg=AZ_GRID_DEG,
        n_sources=N_SOURCES,
    )
    print('estimated peaks:',peaks)

    est_az_deg = np.array([p[0] for p in peaks], dtype=np.float64)

    return {
        "array_name": "Linear ULA (4 mics)",
        "mic_locs": mic_locs,
        "y_mics": y_mics,
        "chosen_bin_hz": chosen_bin_hz,
        "X_bin": X_bin,
        "R": R,
        "P": P,
        "peaks": peaks,
        "est_az_deg": est_az_deg,
    }

def plot_linear_music_1d(
    result: dict,
    true_az_deg: np.ndarray,
):
    P = result["P"]

    plt.figure(figsize=(10, 4))
    plt.plot(
        AZ_GRID_DEG,
        10.0 * np.log10(P + 1e-12),
        label="Linear MUSIC",
    )

    for az in true_az_deg:
        plt.axvline(az, linestyle="--")

    for az, _ in result["peaks"]:
        plt.axvline(az, linestyle=":")

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Normalized spectrum (dB)")
    plt.title(f"{result['array_name']} MUSIC spectrum at {result['chosen_bin_hz']:.2f} Hz")
    plt.xlim(AZ_GRID_DEG[0], AZ_GRID_DEG[-1])
    plt.tight_layout()
    plt.show()

raw_sources, narrow_sources, center_hz = prepare_narrowband_sources(
    LIBRISPEECH_FILES,
    DURATION_SEC,
)
print('true source size:', raw_sources[0].shape) #(3, 48000)
print('true source size:', narrow_sources[0].shape) #(3, 48000)
print('center frequency Hz:', center_hz) #1800Hz

source_positions = build_source_positions() #(3,3)
print(source_positions)

linear_mics, linear_spacing = build_linear_array() #(4,3), 2.667
print(linear_mics)
linear_result = run_linear_music_pipeline(
    mic_locs=linear_mics,
    source_positions=source_positions,
    source_signals=narrow_sources,
    center_hz=center_hz,
)

print("center_hz:", center_hz)
print("linear spacing:", linear_spacing)
print("estimated azimuths:", linear_result["est_az_deg"])
try:
    rmse_val = azimuth_rmse_deg(SOURCE_AZIMUTH_DEG, linear_result["est_az_deg"])
    print("azimuth RMSE:", rmse_val)
except ValueError as e:
    print(f"azimuth RMSE: 계산 불가 ({e})")

plot_linear_music_1d(linear_result, SOURCE_AZIMUTH_DEG)