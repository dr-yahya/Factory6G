"""
Indoor factory MAC-layer simulation.

Translated from main_modified_clean_LTE_final_Athirah.m
Blocks 1 (Mobility), 2 (PHY lookup), and 3 (MAC loop).
(Athirah Mohd Ramly, UKM)

Entry point: run_factory_mac_sim(cfg, data_dir) → dict of per-scenario results.
"""

import os
import logging

import numpy as np
import scipy.io

from src.athirah.mac.path_loss import (
    received_power_dbm,
    noise_power_dbm,
    sinr_dbm as _sinr,
    dist_from_sinr,
    RXSEN_DBM,
    PTX_DBM,
    GT_DB,
    GR_DB,
    NF_DB,
    BW_HZ,
    PLEXP,
)
from src.athirah.mac.sps_scheduling import sps_rssi

logger = logging.getLogger(__name__)

# Factory geometry (from MATLAB Block 1)
FACTORY_LENGTH_M = 1000.0
FACTORY_LANE_WIDTH_M = 15.0
NUM_LANE = 3
LANE_Y_M = np.array([0.0, 15.0, 18.0, 33.0, 36.0, 51.0, 54.0, 69.0])  # m

# Sensor density presets (sensors / lane / km)
DENSITY_PRESETS = {
    "sparse":   25,
    "moderate": 50,
    "dense":    100,
}

# SPS parameters (from MATLAB Block 3)
KSI     = 0      # half-duplex
RES     = 1.0    # all V2V resources
D_TH    = 360.0  # m — receiver distance threshold (moving scenario)
D_SENSE = 100.0  # m — sensing range


def _place_sensors(num_sen_per_lane: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Poisson-distributed sensor placement along factory lanes.

    Translated from Block 1 of main_modified_clean_LTE_final_Athirah.m.

    Returns:
        x_loc: shape (NUM_LANE * num_sen_per_lane,) — x positions (m)
        y_loc: shape (NUM_LANE * num_sen_per_lane,) — y positions (m)
    """
    mean_inter_d = FACTORY_LENGTH_M / num_sen_per_lane
    # Poisson inter-sensor distances, cumulative sum mod FactoryLength
    x_all, y_all = [], []
    for lane_idx in range(NUM_LANE):
        gaps = rng.poisson(mean_inter_d, size=num_sen_per_lane)
        x_locs = np.mod(np.cumsum(gaps), FACTORY_LENGTH_M)
        x_locs = np.sort(x_locs)
        y_locs = np.full(num_sen_per_lane, LANE_Y_M[lane_idx])
        x_all.append(x_locs)
        y_all.append(y_locs)
    return np.concatenate(x_all), np.concatenate(y_all)


def run_factory_mac_sim(cfg: dict, data_dir: str) -> dict:
    """
    Run MAC-layer factory simulation.

    Translated from Blocks 1–3 of main_modified_clean_LTE_final_Athirah.m.

    Args:
        cfg:      Full config dict; reads cfg['jidd_scma'] section.
        data_dir: Path to 'dr_athirah_simulation/MAC and APP layer/' directory
                  containing the .mat lookup-table files.

    Returns:
        results dict keyed by (density, speed_kmh, fc_hz) with fields:
            dist_all, sinr_all, per_acn_all, per_app_all, per_mac_all,
            prr_uncoded, e2edelay_uncoded, prr_app, e2edelay_app,
            prr_mac, e2edelay_mac
    """
    mac_cfg = cfg.get("jidd_scma", {})
    densities  = mac_cfg.get("densities", ["sparse", "moderate", "dense"])
    speeds_kmh = mac_cfg.get("speeds_kmh", [0, 3, 7, 10])
    freqs_hz   = [f * 1e9 for f in mac_cfg.get("carrier_frequencies_ghz", [3.5, 28.0])]
    seed       = cfg.get("simulation", {}).get("seed", 42)

    # Load PHY PER lookup tables (moving scenario)
    snr_phy_path = os.path.join(data_dir, "SNR_PHY_Linear_WO_Div.mat")
    per_phy_path = os.path.join(data_dir, "PER_PHY_Linear_WO_Div.mat")
    snr_table = scipy.io.loadmat(snr_phy_path)["EbN0"].flatten().astype(float)
    per_table = scipy.io.loadmat(per_phy_path)["PER"].flatten().astype(float)

    results = {}
    rng = np.random.default_rng(seed)

    pkt_sizes = [50, 100, 300, 500]  # bytes — same as MATLAB

    noiseP = noise_power_dbm(BW_HZ, NF_DB)

    for density_name in densities:
        num_sen_per_lane = DENSITY_PRESETS[density_name]
        num_sen_per_lane_km = num_sen_per_lane  # already per km per lane
        sen_density_total = num_sen_per_lane * NUM_LANE
        sen_density_km    = num_sen_per_lane_km / 1000.0 * NUM_LANE

        for v_kmh in speeds_kmh:
            for fc_hz in freqs_hz:
                plexp = PLEXP.get(fc_hz, 3.6)
                lam = 3e8 / fc_hz

                logger.info(
                    f"[factory_mac] density={density_name}, v={v_kmh}km/h, "
                    f"fc={fc_hz/1e9:.1f}GHz, N={sen_density_total}"
                )

                # ── Block 1: Sensor placement ──────────────────────────────
                x_loc, y_loc = _place_sensors(num_sen_per_lane, rng)
                N = len(x_loc)  # = sen_density_total

                # ── Block 2: PHY — pairwise distance → SNR → PER ──────────
                # Compute pairwise distances (N × N)
                dx = x_loc[:, None] - x_loc[None, :]  # (N, N)
                dy = y_loc[:, None] - y_loc[None, :]
                dist_mat = np.sqrt(dx**2 + dy**2)  # (N, N)
                np.fill_diagonal(dist_mat, 1e-3)    # avoid zero distance

                # Received power for each pair (N × N)
                prx_mat = received_power_dbm(dist_mat, fc_hz=fc_hz,
                                             ptx_dbm=PTX_DBM, gt_db=GT_DB,
                                             gr_db=GR_DB, plexp=plexp)  # (N, N)

                # Per-sensor interference index: sensors above RxSen threshold
                intf_mask = prx_mat > RXSEN_DBM  # (N, N)

                # SNR for each (sensor, receiver) pair
                snr_mat = np.zeros((N, N), dtype=float)
                for k in range(N):
                    intf_idx = np.where(intf_mask[k])[0]
                    intf_prx = prx_mat[k, intf_idx]
                    for kk in range(N):
                        snr_mat[k, kk] = _sinr(
                            prx_mat[k, kk], noiseP, intf_prx,
                            ksi=KSI, ptx_dbm=PTX_DBM
                        )

                # PER lookup from PHY table for each packet size
                def lookup_per(snr_val: float) -> float:
                    idx = int(np.argmin(np.abs(snr_table - snr_val)))
                    return float(per_table[idx])

                # We use 4 packet sizes but store per scenario:
                # pktsize 500 = acn (no FEC), 300 = cam, 100 = app, 50 = mac
                per_acn_mat = np.vectorize(lookup_per)(snr_mat)
                per_cam_mat = per_acn_mat.copy()  # same PER table for all sizes
                per_app_mat = per_acn_mat.copy()
                per_mac_mat = per_acn_mat.copy()

                # ── Block 3: MAC — SPS scheduling per sensor ──────────────
                prr_uncoded   = np.zeros((N, N))
                e2e_uncoded   = np.zeros((N, N))
                prr_app_arr   = np.zeros((N, N))
                e2e_app_arr   = np.zeros((N, N))
                prr_mac_arr   = np.zeros((N, N))
                e2e_mac_arr   = np.zeros((N, N))

                for k in range(N):
                    dist_k = dist_mat[k]
                    pwr_k  = prx_mat[k]
                    intf_k = np.where(intf_mask[k])[0]

                    for kk in range(N):
                        pe_cam = per_cam_mat[k]  # shape (N,) — CAM PER for all sensors

                        # Uncoded (500-byte ACN)
                        prr_val, delay_val, _ = sps_rssi(
                            pwr_k, dist_k, intf_k,
                            psize=500, pe_cam=pe_cam,
                            pe_acn=float(per_acn_mat[k, kk]),
                            sen_density_km=sen_density_km,
                            v=float(v_kmh), K=1,
                            d_th=D_TH, ksi=KSI, res=RES, d_sense=D_SENSE
                        )
                        prr_uncoded[k, kk] = prr_val
                        e2e_uncoded[k, kk] = delay_val

                        # APP-layer FEC (200-byte, K=8 source symbols)
                        K_src = cfg.get("jidd_scma", {}).get("raptor_q_K", 8)
                        prr_val2, delay_val2, _ = sps_rssi(
                            pwr_k, dist_k, intf_k,
                            psize=200, pe_cam=pe_cam,
                            pe_acn=float(per_app_mat[k, kk]),
                            sen_density_km=sen_density_km,
                            v=float(v_kmh), K=K_src,
                            d_th=D_TH, ksi=KSI, res=RES, d_sense=D_SENSE
                        )
                        prr_app_arr[k, kk] = prr_val2
                        e2e_app_arr[k, kk] = delay_val2

                        # MAC-layer FEC (50-byte, K=8)
                        prr_val3, delay_val3, _ = sps_rssi(
                            pwr_k, dist_k, intf_k,
                            psize=50, pe_cam=pe_cam,
                            pe_acn=float(per_mac_mat[k, kk]),
                            sen_density_km=sen_density_km,
                            v=float(v_kmh), K=K_src,
                            d_th=D_TH, ksi=KSI, res=RES, d_sense=D_SENSE
                        )
                        prr_mac_arr[k, kk] = prr_val3
                        e2e_mac_arr[k, kk] = delay_val3

                # Aggregate: dist and SNR (flatten all sensor pairs)
                dist_all = dist_mat.flatten()
                sinr_all = snr_mat.flatten()

                results[(density_name, float(v_kmh), fc_hz)] = {
                    "dist_all":       dist_all,
                    "sinr_all":       sinr_all,
                    "per_acn_all":    per_acn_mat.flatten(),
                    "per_app_all":    per_app_mat.flatten(),
                    "per_mac_all":    per_mac_mat.flatten(),
                    "prr_uncoded":    prr_uncoded.flatten(),
                    "e2edelay_uncoded": e2e_uncoded.flatten(),
                    "prr_app":        prr_app_arr.flatten(),
                    "e2edelay_app":   e2e_app_arr.flatten(),
                    "prr_mac":        prr_mac_arr.flatten(),
                    "e2edelay_mac":   e2e_mac_arr.flatten(),
                    # Distance estimated from SINR (for paper-style plots)
                    "dist_from_sinr": dist_from_sinr(
                        sinr_all, noiseP, fc_hz=fc_hz,
                        ptx_dbm=PTX_DBM, gt_db=GT_DB, gr_db=GR_DB, plexp=plexp
                    ),
                }

                logger.info(
                    f"[factory_mac] done: mean_prr_uncoded="
                    f"{prr_uncoded.mean():.3f}"
                )

    return results
