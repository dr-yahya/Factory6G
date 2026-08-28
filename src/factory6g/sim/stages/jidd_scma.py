"""
JIDD-SCMA simulation stage wrapper.

Runs Dr. Athirah's polar-coded SCMA-OFDM simulation and formats results
to match the existing stage_result schema used by write_stage_outputs().
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factory6g.sim.stages.common import MIN_RESOLVED_BIT_ERRORS, POINT_STATUS_RESOLVED, POINT_STATUS_UPPER_BOUND_ONLY


# Path to Dr. Athirah's MATLAB reference data files (repo_root/reference/dr_athirah_simulation/)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_REFERENCE_DIR = _REPO_ROOT / "reference" / "dr_athirah_simulation"
_DATA_DIR = str(_REFERENCE_DIR / "PHY layer")
_MAC_APP_DIR = str(_REFERENCE_DIR / "MAC and APP layer")


def run_jidd_scma_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Run the JIDD-SCMA BER simulation and return a stage_result dict.

    Args:
        cfg: Full config dict (reads cfg["jidd_scma"] section).

    Returns:
        stage_result dict compatible with write_stage_outputs():
            {
                "stage": "jidd_scma",
                "ebno_db_range": [...],
                "methods": {"jidd_scma": {metric: [values...], ...}},
                "runtime_totals_sec": {"jidd_scma": float},
            }
    """
    from factory6g.athirah.simulation import run_jidd_scma

    jidd_cfg = cfg.get("jidd_scma", {})
    ebno_start = jidd_cfg.get("ebno_db_range", [0, 10])[0]
    ebno_stop  = jidd_cfg.get("ebno_db_range", [0, 10])[1]
    ebno_step  = jidd_cfg.get("ebno_db_step", 1)

    import numpy as np
    ebno_db_range = list(np.arange(ebno_start, ebno_stop + 1e-9, ebno_step))

    print("[jidd_scma] Starting JIDD-SCMA simulation ...")
    print(f"[jidd_scma] Eb/N0 range: {ebno_start}–{ebno_stop} dB, step {ebno_step} dB")
    print(f"[jidd_scma] Data dir: {_DATA_DIR}")

    ce_method = jidd_cfg.get("ce_method", "mmse")
    print(f"[jidd_scma] CE method: {ce_method}")

    t0 = time.perf_counter()
    raw_results = run_jidd_scma(jidd_cfg, _DATA_DIR, ce_method=ce_method)
    total_runtime = time.perf_counter() - t0

    # Format per-metric lists in Eb/N0 order
    ber_list      = []
    bit_err_list  = []
    total_bits_list = []
    runtime_list  = []
    status_list   = []

    for ebn0 in ebno_db_range:
        point = raw_results.get(float(ebn0), {})
        ber        = float(point.get("ber", float("nan")))
        bit_errors = float(point.get("bit_errors", 0))
        total_bits = float(point.get("total_bits", 0))
        rt         = float(point.get("runtime_sec", 0.0))

        ber_list.append(ber)
        bit_err_list.append(bit_errors)
        total_bits_list.append(total_bits)
        runtime_list.append(rt)

        status = (
            POINT_STATUS_RESOLVED
            if bit_errors >= MIN_RESOLVED_BIT_ERRORS
            else POINT_STATUS_UPPER_BOUND_ONLY
        )
        status_list.append(status)

    stage_result: dict[str, Any] = {
        "stage": "jidd_scma",
        "ebno_db_range": [float(e) for e in ebno_db_range],
        "methods": {
            "jidd_scma": {
                "ber": ber_list,
                "ber_upper_confidence": ber_list,   # no CI estimate — use same BER
                "bit_errors": bit_err_list,
                "total_bits": total_bits_list,
                "runtime_sec": runtime_list,
                "point_status": status_list,
            }
        },
        "runtime_totals_sec": {"jidd_scma": total_runtime},
    }

    # ── Optional MAC layer simulation ─────────────────────────────────────────
    if jidd_cfg.get("mac_enabled", False):
        from factory6g.athirah.mac.factory_sim import run_factory_mac_sim
        print("[jidd_scma] Running MAC layer simulation ...")
        t_mac = time.perf_counter()
        mac_results = run_factory_mac_sim(cfg, _MAC_APP_DIR)
        stage_result["mac_metrics"] = mac_results
        stage_result["runtime_totals_sec"]["mac"] = time.perf_counter() - t_mac
        print(f"[jidd_scma] MAC done in {stage_result['runtime_totals_sec']['mac']:.1f}s")

    # ── Optional APP layer simulation ─────────────────────────────────────────
    if jidd_cfg.get("app_enabled", False):
        from factory6g.athirah.app.e2e_metrics import run_app_fec_sim
        print("[jidd_scma] Running APP layer FEC simulation ...")
        t_app = time.perf_counter()
        # MAC results must exist; run MAC first if not already done
        if "mac_metrics" not in stage_result:
            from factory6g.athirah.mac.factory_sim import run_factory_mac_sim
            mac_results = run_factory_mac_sim(cfg, _MAC_APP_DIR)
        else:
            mac_results = stage_result["mac_metrics"]
        app_results = run_app_fec_sim(mac_results, _MAC_APP_DIR)
        stage_result["app_metrics"] = app_results
        stage_result["runtime_totals_sec"]["app"] = time.perf_counter() - t_app
        print(f"[jidd_scma] APP done in {stage_result['runtime_totals_sec']['app']:.1f}s")

    return stage_result
