"""
Factory path-loss and received-power models.

Translated from pathLossModel.m and the PHY block of
main_modified_clean_LTE_final_Athirah.m (Athirah Mohd Ramly, UKM).

The main simulation uses the log-distance model with:
  Prx_dBm = Ptx + Gt + Gr + 20*log10(lambda/(4*pi)) - PLexp*10*log10(dist)
where PLexp = 3.6 for 3.5 GHz, 3.1 for 28 GHz (Paper 1, Table 2).
"""

import numpy as np


# Default RF parameters (from main_modified_clean_LTE_final_Athirah.m)
_C = 3e8          # speed of light (m/s)
PTX_DBM = 29.0    # transmit power (dBm)
GT_DB   = 0.0     # transmit antenna gain (dB)
GR_DB   = 8.0     # receive antenna gain (dB)
NF_DB   = 7.0     # noise figure (dB)
BW_HZ   = 0.22e6  # bandwidth (Hz) — 3.5 GHz channel
RXSEN_DBM = -105.4  # receiver sensitivity (dBm)
THERMAL_NOISE_DBM_HZ = -174.0  # kTB at 290 K (dBm/Hz)

# Path-loss exponents per carrier frequency (Paper 1, Table 2)
PLEXP = {
    3.5e9: 3.6,
    28e9:  3.1,
}


def noise_power_dbm(bw_hz: float = BW_HZ, nf_db: float = NF_DB) -> float:
    """
    Thermal noise power in dBm.

    noiseP = thermalNoise + 10*log10(BW) + NF
    """
    return THERMAL_NOISE_DBM_HZ + 10.0 * np.log10(bw_hz) + nf_db


def wavelength(fc_hz: float) -> float:
    """Free-space wavelength (m)."""
    return _C / fc_hz


def received_power_dbm(
    dist_m: np.ndarray,
    fc_hz: float = 3.5e9,
    ptx_dbm: float = PTX_DBM,
    gt_db: float   = GT_DB,
    gr_db: float   = GR_DB,
    plexp: float | None = None,
) -> np.ndarray:
    """
    Received power (dBm) using log-distance path-loss.

    Prx = Ptx + Gt + Gr + 20*log10(lambda/(4*pi)) - PLexp*10*log10(dist)

    Translated from the PHY block of main_modified_clean_LTE_final_Athirah.m:
        sensor(k).Prx(kk,t) = Ptx + Gt + Gr
                               + 20*log10(lambda/4/pi)
                               + 10*log10((1/sensor(k).dist(kk,t))^PLexp)

    Args:
        dist_m: Distance(s) in metres (scalar or array). Must be > 0.
        fc_hz:  Carrier frequency in Hz.
        ptx_dbm, gt_db, gr_db: TX power and antenna gains (dB/dBm).
        plexp:  Path-loss exponent; if None, uses PLEXP dict (default per fc).

    Returns:
        Prx in dBm, same shape as dist_m.
    """
    dist_m = np.asarray(dist_m, dtype=float)
    dist_m = np.where(dist_m <= 0, 1e-3, dist_m)  # guard against zero

    if plexp is None:
        plexp = PLEXP.get(fc_hz, 3.6)

    lam = wavelength(fc_hz)
    fspl_offset_db = 20.0 * np.log10(lam / (4.0 * np.pi))
    path_loss_db   = plexp * 10.0 * np.log10(dist_m)

    return ptx_dbm + gt_db + gr_db + fspl_offset_db - path_loss_db


def sinr_dbm(
    prx_desired_dbm: float,
    noise_power_dbm: float,
    interference_prx_dbm: np.ndarray,
    ksi: float = 0.0,
    ptx_dbm: float = PTX_DBM,
) -> float:
    """
    SINR in dB.

    Translated from main_modified_clean_LTE_final_Athirah.m:
        SNR(kk) = Prx(kk) - 10*log10(10^(noiseP+ksi*Ptx)/10 + 10^(sum(Prx(intf))/10))

    Args:
        prx_desired_dbm:      Received power of the desired signal (dBm).
        noise_power_dbm:      Thermal noise power (dBm).
        interference_prx_dbm: Array of received powers from interferers (dBm).
        ksi:  Full-duplex flag (1) or half-duplex (0).
        ptx_dbm: TX power (dBm), used only when ksi=1.

    Returns:
        SINR in dB.
    """
    noise_lin = 10.0 ** ((noise_power_dbm + ksi * ptx_dbm) / 10.0)
    if len(interference_prx_dbm) > 0:
        intf_lin = np.sum(10.0 ** (np.asarray(interference_prx_dbm) / 10.0))
    else:
        intf_lin = 0.0
    total_interference_lin = noise_lin + intf_lin
    sinr_db = prx_desired_dbm - 10.0 * np.log10(total_interference_lin)
    return sinr_db


def dist_from_sinr(
    sinr_db: np.ndarray,
    noise_power_dbm: float,
    fc_hz: float = 3.5e9,
    ptx_dbm: float = PTX_DBM,
    gt_db: float   = GT_DB,
    gr_db: float   = GR_DB,
    plexp: float | None = None,
) -> np.ndarray:
    """
    Inverse path-loss: distance from SINR (for plotting).

    Translated from the plot section of main_modified_clean_LTE_final_Athirah.m:
        dist_allN = pathLossModel(par, Ptx, (SNR_all + noiseP), PLexp)
    which is the log-distance inverse:
        dist = (10^((Pt - Pr) / 10) * (lambda/(4*pi))^2)^(1/PLexp)

    Args:
        sinr_db: SINR values (dB), array.
        noise_power_dbm: Noise power (dBm) added to SINR to recover Prx.
        fc_hz, ptx_dbm, gt_db, gr_db, plexp: same as received_power_dbm.

    Returns:
        Estimated distance in metres.
    """
    sinr_db = np.asarray(sinr_db, dtype=float)
    prx_dbm = sinr_db + noise_power_dbm  # approximate: Pr ≈ SNR + N0 when no interference

    if plexp is None:
        plexp = PLEXP.get(fc_hz, 3.6)

    lam = wavelength(fc_hz)
    fspl_ref = (lam / (4.0 * np.pi)) ** 2
    power_ratio_lin = 10.0 ** ((ptx_dbm + gt_db + gr_db - prx_dbm) / 10.0)

    dist_m = (power_ratio_lin * fspl_ref) ** (1.0 / plexp)
    return dist_m
