"""
Semi-Persistent Scheduling (SPS) with RSSI-based resource selection.

Translated from sps_rssi_mod_new_2.m (Athirah Mohd Ramly, UKM).

Computes PRR and MAC delay for a given sensor given its power/distance
environment, implementing Algorithm 1 from Paper 1.
"""

import math
import numpy as np


def sps_rssi(
    pwr: np.ndarray,
    dist: np.ndarray,
    intf: np.ndarray,
    psize: float,
    pe_cam: np.ndarray,
    pe_acn: float,
    sen_density_km: float,
    v: float,
    K: float,
    d_th: float,
    ksi: int = 0,
    res: float = 1.0,
    d_sense: float = 100.0,
) -> tuple[float, float, int]:
    """
    SPS RSSI-based scheduling PRR and MAC E2E delay.

    Translated from sps_rssi_mod_new_2.m.

    Args:
        pwr:             Received power array for all sensors (dBm), shape (N,).
        dist:            Distance array from this sensor to all others (m), shape (N,).
        intf:            Indices of interfering sensors (0-based).
        psize:           Packet size (bytes).
        pe_cam:          PER array for CAM (background) messages, shape (N,).
        pe_acn:          PER for the ACN (event) message of this link.
        sen_density_km:  Sensor density per km (sensors/km).
        v:               Speed (km/h).
        K:               Number of source symbols (1 for uncoded).
        d_th:            Distance threshold for intended receivers (m).
        ksi:             0 = half-duplex, 1 = full-duplex.
        res:             Resource fraction (1.0 = all resources available).
        d_sense:         Sensing range (m).

    Returns:
        (prr, macdelay_ms, max_rx)
        prr:         Packet reception rate.
        macdelay_ms: MAC-layer E2E delay (ms).
        max_rx:      Maximum number of simultaneous receivers.
    """
    SC_RB = 4 * res                             # subchannels per resource block
    Num_RB_DENM = math.ceil((psize * 8) / 108)  # 108 = 12 subcarriers × 9 slots
    Num_RB_CAM  = math.ceil((8 * 300) / 108)
    Num_SC_DENM = math.ceil(Num_RB_DENM / SC_RB)
    Num_SC_CAM  = math.ceil(Num_RB_CAM  / SC_RB)
    CAM_time    = Num_SC_CAM  * 0.1  # ms
    DENM_time   = Num_SC_DENM * 0.1  # ms

    PRB_data = 40 * res
    if ksi == 0:
        Nb_CAM  = math.floor(100 / math.ceil(Num_RB_CAM  / PRB_data))
        Nb_DENM = math.floor(100 / math.ceil(Num_RB_DENM / PRB_data))
    else:
        Nb_CAM  = 10 * math.floor((PRB_data * 10) / Num_RB_CAM)
        Nb_DENM = 10 * math.floor((PRB_data * 10) / Num_RB_DENM)

    # Interfering sensors within sensing range
    dist_arr = np.asarray(dist, dtype=float)
    intf_arr = np.asarray(intf, dtype=int)
    Tx = int(np.sum(dist_arr[intf_arr] < d_sense)) if len(intf_arr) > 0 else 0

    # Available receive slots
    if ksi == 0:
        CAM_total_time  = Tx * math.ceil(CAM_time)
        DENM_total_time = math.ceil(DENM_time)
        max_rx_time     = math.floor(100 - (DENM_total_time + CAM_total_time))
        max_rx          = math.floor(max_rx_time / math.ceil(DENM_time))
    else:
        CAM_total_time  = Tx * CAM_time
        max_rx_time     = math.floor(100 - (DENM_time + CAM_total_time))
        max_rx          = math.floor(max_rx_time / DENM_time)

    # Intended receivers within d_th
    Rx = int(np.sum(dist_arr < d_th))

    if max_rx > Rx:
        tao_acn = 1.0
    elif max_rx < 0:
        tao_acn = 0.0
    else:
        tao_acn = max_rx / Rx if Rx > 0 else 0.0

    prr      = tao_acn * (1.0 - pe_acn)
    macdelay = DENM_time  # ms

    return prr, macdelay, max_rx
