import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from pathlib import Path

# ────────────────────────────────────────────────────────────
#  PARAMÈTRES À ADAPTER ──────────────────────────────────────
# ────────────────────────────────────────────────────────────
COMBINED_PKL = (
    "C:/Users/Ethier Lab/Documents/GitHub/Decoder-Processing/"
    "DataSET/Jango_ISO_2015/combined.pkl"
)
CHANNEL          = 0      # index du canal EMG à visualiser
SECONDS_TO_PLOT  = 10     # durée de la fenêtre affichée (s)
BIN_FACTOR       = 20     # 1 ms → 20 ms  ⇒ 1000 Hz → 50 Hz
FS_RAW           = 1000.0 # Hz – fréquence d'échantillonnage brute
FC               = 5.0    # Hz – fréquence de coupure du passe‑bas
ORDER            = 4      # ordre du filtre
# ────────────────────────────────────────────────────────────

def butter_lowpass(data, fs=FS_RAW, fc=FC, order=ORDER):
    """Passe‑bas Butterworth."""
    b, a = butter(order, fc / (fs / 2), "low")
    return filtfilt(b, a, data, axis=0)

def downsample_mean(arr, factor=BIN_FACTOR):
    """Moyenne par blocs (décimation sans anti‑repliement)."""
    keep = (len(arr) // factor) * factor
    return arr[:keep].reshape(-1, factor).mean(axis=1)

def main():
    # ── lecture du day‑0 ─────────────────────────────────────
    df = pd.read_pickle(COMBINED_PKL)
    day0 = df[df["date"] == df["date"].min()].iloc[0]
    emg_df = day0["EMG"]          # DataFrame (n_time, n_channels)
    emg = emg_df.values[:, CHANNEL]

    # ── traitements ─────────────────────────────────────────
    rect_raw = np.abs(emg)                            # rectification
    filt_raw = butter_lowpass(rect_raw)               # filtre 5 Hz

    rect_ds  = downsample_mean(rect_raw)              # ↓ vers 50 Hz
    # deux stratégies :
    filt_raw_ds = downsample_mean(filt_raw)           # filtre → DS
    rect_ds_flt = butter_lowpass(rect_ds, fs=FS_RAW / BIN_FACTOR)

    fs_ds = FS_RAW / BIN_FACTOR                       # 50 Hz
    t_raw = np.arange(len(rect_raw)) / FS_RAW
    t_ds  = np.arange(len(rect_ds))  / fs_ds

    # ── FIGURE 1 – brute 1000 Hz ───────────────────────────
    plt.figure(figsize=(9, 3))
    plt.plot(t_raw, rect_raw, label="Rectifiée brute", alpha=0.4)
    plt.plot(t_raw, filt_raw, label="Filtrée 5 Hz", lw=2)
    plt.xlim(0, SECONDS_TO_PLOT)
    plt.xlabel("Temps (s)"); plt.ylabel("Amplitude (a.u.)")
    plt.ylim(-0.5, 10)
    # plt.title("EMG rectifié : brut vs. passe‑bas 5 Hz")
    
    plt.legend(); plt.tight_layout()
    plt.savefig("emg_rectif.png", dpi=700)
    # ── FIGURE 2 – comparatif 50 Hz ────────────────────────
    plt.figure(figsize=(9, 3))
    plt.plot(t_ds, rect_ds,      label="Rectifiée brute ", alpha=0.4)
    plt.plot(t_ds, filt_raw_ds,  label="Filtre→DS", lw=2)
    plt.plot(t_ds, rect_ds_flt,  label="DS→filtre", lw=1.5, ls="--")
    plt.xlim(0, SECONDS_TO_PLOT)
    plt.xlabel("Temps (s)"); plt.ylabel("Amplitude (a.u.)")
    # plt.title("Comparatif après down‑sampling (50 Hz)")
    plt.legend(); plt.tight_layout()

    # ── FIGURE 3 – PSD (log‑scale) ─────────────────────────
    freqs, P_rect  = welch(rect_ds,      fs=fs_ds, nperseg=1024)
    _,     P_fraw = welch(filt_raw_ds,  fs=fs_ds, nperseg=1024)
    _,     P_fds  = welch(rect_ds_flt,  fs=fs_ds, nperseg=1024)

    plt.figure(figsize=(6, 4))
    plt.semilogy(freqs, P_rect,  label="Rectifié (50 Hz)")
    plt.semilogy(freqs, P_fraw,  label="Filtre→DS")
    plt.semilogy(freqs, P_fds,   label="DS→filtre")
    plt.xlim(0, 25); plt.xlabel("Fréquence (Hz)")
    plt.ylabel("PSD (a.u./Hz)")
    # plt.title("Densité spectrale de puissance")
    plt.legend(); plt.tight_layout()
    plt.savefig("emg_spect.png", dpi=700)
    plt.show()

# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not Path(COMBINED_PKL).exists():
        raise FileNotFoundError(
            f"Impossible de trouver {COMBINED_PKL}\n"
            "→ vérifiez le chemin dans le bloc PARAMÈTRES."
        )
    main()