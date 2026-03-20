"""
End-to-end (E2E) performance metrics for APP-layer FEC schemes.

Implements Eq. 11-14 from Paper 2 (Ramly et al. 2022).
Used after factory_sim.py produces MAC-layer PRR and E2E delay.
"""

import numpy as np


def e2e_delay_repetition(
    overhead: float,
    mac_delay_ms: float,
) -> float:
    """
    E2E delay for repetition codes (Paper 2, Eq. 12).

    e2edelay_repAPP = overhead * e2edelay_uncoded

    Translated from Block 4 of main_modified_clean_LTE_final_Athirah.m.

    Args:
        overhead:      Average transmissions needed (from repetition table).
        mac_delay_ms:  MAC E2E delay without FEC (ms).

    Returns:
        E2E delay in ms.
    """
    return overhead * mac_delay_ms


def e2e_delay_raptor_q(
    overhead: float,
    mac_delay_ms: float,
) -> float:
    """
    E2E delay for Raptor-Q codes (Paper 2, Eq. 13).

    e2edelay_rqAPP = overhead * e2edelay_APP
    where overhead = K / CR_optimal.

    Translated from Block 4 of main_modified_clean_LTE_final_Athirah.m.

    Args:
        overhead:      K / CR — Raptor-Q encoding overhead.
        mac_delay_ms:  MAC E2E delay for the APP-layer packet size (ms).

    Returns:
        E2E delay in ms.
    """
    return overhead * mac_delay_ms


def prr_combined(
    prr_mac: float,
    prr_fec: float,
) -> float:
    """
    Combined packet reception rate (Paper 2, Eq. 11).

    PRR_total = PRR_MAC * PRR_FEC

    Args:
        prr_mac: MAC-layer PRR.
        prr_fec: FEC (APP or MAC layer) PRR.

    Returns:
        Combined PRR.
    """
    return prr_mac * prr_fec


def run_app_fec_sim(
    mac_results: dict,
    mat_dir: str,
    target_per: float = 0.01,
) -> dict:
    """
    Run APP-layer FEC sweep over all (density, speed, fc) scenarios.

    Calls Raptor-Q and repetition code lookup for every sensor pair in
    each scenario from factory_sim results, then computes PRR and E2E delay.

    Args:
        mac_results: Output from run_factory_mac_sim().
        mat_dir:     Directory containing the .mat lookup tables.
        target_per:  Target PER for code rate selection (default 0.01).

    Returns:
        Dict with same keys as mac_results, each value extended with:
            prr_rep_app, e2edelay_rep_app, overh_rep_app,
            prr_rq_app,  e2edelay_rq_app,  overh_rq_app,
            prr_rq_mac,  e2edelay_rq_mac,  overh_rq_mac
    """
    from src.athirah.app.raptor_q import load_raptor_q_table, raptor_q_prr_overhead
    from src.athirah.app.repetition import load_repetition_table, repetition_prr_overhead

    # Load tables once
    mat_rq, per_axis, cr_axis = load_raptor_q_table(mat_dir)
    prr_rep, overh_rep, ref_axis = load_repetition_table(mat_dir)

    app_results = {}
    for key, res in mac_results.items():
        N = len(res["prr_uncoded"])

        prr_rep_app    = np.zeros(N)
        e2e_rep_app    = np.zeros(N)
        overh_rep_app  = np.zeros(N)

        prr_rq_app     = np.zeros(N)
        e2e_rq_app     = np.zeros(N)
        overh_rq_app   = np.zeros(N)

        prr_rq_mac     = np.zeros(N)
        e2e_rq_mac     = np.zeros(N)
        overh_rq_mac   = np.zeros(N)

        for i in range(N):
            # Repetition code (uncoded PRR → rep table)
            p_rep, oh_rep = repetition_prr_overhead(
                float(res["prr_uncoded"][i]), prr_rep, overh_rep, ref_axis, target_per
            )
            prr_rep_app[i]   = p_rep
            overh_rep_app[i] = oh_rep
            e2e_rep_app[i]   = e2e_delay_repetition(oh_rep, float(res["e2edelay_uncoded"][i]))

            # Raptor-Q at APP layer
            p_rq_app, oh_rq_app, _ = raptor_q_prr_overhead(
                float(res["prr_app"][i]), mat_rq, per_axis, cr_axis, target_per
            )
            prr_rq_app[i]   = p_rq_app
            overh_rq_app[i] = oh_rq_app
            e2e_rq_app[i]   = e2e_delay_raptor_q(oh_rq_app, float(res["e2edelay_app"][i]))

            # Raptor-Q at MAC layer
            p_rq_mac, oh_rq_mac, _ = raptor_q_prr_overhead(
                float(res["prr_mac"][i]), mat_rq, per_axis, cr_axis, target_per
            )
            prr_rq_mac[i]   = p_rq_mac
            overh_rq_mac[i] = oh_rq_mac
            e2e_rq_mac[i]   = e2e_delay_raptor_q(oh_rq_mac, float(res["e2edelay_mac"][i]))

        app_results[key] = {
            **res,
            "prr_rep_app":   prr_rep_app,
            "e2edelay_rep_app": e2e_rep_app,
            "overh_rep_app": overh_rep_app,
            "prr_rq_app":    prr_rq_app,
            "e2edelay_rq_app": e2e_rq_app,
            "overh_rq_app":  overh_rq_app,
            "prr_rq_mac":    prr_rq_mac,
            "e2edelay_rq_mac": e2e_rq_mac,
            "overh_rq_mac":  overh_rq_mac,
        }

    return app_results
