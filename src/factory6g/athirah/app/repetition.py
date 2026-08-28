"""
Repetition-code APP-layer FEC performance via pre-computed lookup table.

Translated from fec_repetition_new.m and Block 4 of
main_modified_clean_LTE_final_Athirah.m (Athirah Mohd Ramly, UKM).

The MATLAB generates 'rep_iter4_K1_Table_prr_overh.mat' by Monte-Carlo
simulation (Nr=4 repetitions, K=1 source packet). This Python module loads
that pre-computed table and performs the same table lookup.
"""

import os
import numpy as np
import scipy.io


def load_repetition_table(mat_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the repetition-code PRR/overhead lookup table.

    MATLAB: load('rep_iter4_K1_Table_prr_overh.mat')
    Variables: prr_repS, overh_repS
    Axis:      ref = [1:-0.05:0.05, 0.01]

    Args:
        mat_dir: Directory containing 'rep_iter4_K1_Table_prr_overh.mat'.

    Returns:
        (prr_rep, overh_rep, ref_axis)
        prr_rep:   PRR values, array of length n.
        overh_rep: Overhead values, array of length n.
        ref_axis:  Input PRR axis (unencoded PRR), array of length n.
    """
    fpath = os.path.join(mat_dir, "rep_iter4_K1_Table_prr_overh.mat")
    data = scipy.io.loadmat(fpath)
    prr_rep   = data["prr_repS"].flatten().astype(float)
    overh_rep = data["overh_repS"].flatten().astype(float)

    # Reference PRR axis: ref = [1:-0.05:0.05, 0.01]
    ref_axis = np.concatenate([np.arange(1.0, 0.0, -0.05), [0.01]])
    # Trim to length of table
    n = min(len(prr_rep), len(ref_axis))
    return prr_rep[:n], overh_rep[:n], ref_axis[:n]


def repetition_prr_overhead(
    prr_uncoded: float,
    prr_rep: np.ndarray,
    overh_rep: np.ndarray,
    ref_axis: np.ndarray,
    target_prr: float = 0.01,
) -> tuple[float, float]:
    """
    Repetition-code APP PRR and overhead via table lookup.

    Translated from Block 4 of main_modified_clean_LTE_final_Athirah.m:
        [x,idxR] = min(abs(sensor(k).prr_uncoded(kk,t)-ref));
        if sensor(k).prr_uncoded(kk,t) < target
            prr_repAPP = 0;
        else
            prr_repAPP = prr_repS(idxR);
        end
        overh_repAPP = overh_repS(idxR);
        e2edelay_repAPP = overh_repAPP * e2edelay_uncoded;

    Args:
        prr_uncoded: Unencoded (uncoded) PHY PRR for this link.
        prr_rep:     PRR lookup table from load_repetition_table.
        overh_rep:   Overhead lookup table from load_repetition_table.
        ref_axis:    Input PRR axis from load_repetition_table.
        target_prr:  Minimum viable PRR (default 0.01); below this, output PRR=0.

    Returns:
        (prr_out, overhead)
        prr_out:  Repetition-coded PRR.
        overhead: Average number of transmissions needed (overhead factor).
    """
    if prr_uncoded < target_prr:
        return 0.0, float(overh_rep[-1])

    idx = int(np.argmin(np.abs(ref_axis - prr_uncoded)))
    return float(prr_rep[idx]), float(overh_rep[idx])
