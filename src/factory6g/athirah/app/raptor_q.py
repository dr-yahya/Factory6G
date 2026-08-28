"""
Raptor-Q APP-layer FEC performance via pre-computed lookup table.

Translated from Block 4 of main_modified_clean_LTE_final_Athirah.m
(Athirah Mohd Ramly, UKM).

The MATLAB implementation does NOT encode/decode Raptor-Q directly;
it loads a pre-computed lookup table from 'K8_T512_Table_CR_PER_full.mat'
and looks up PER at a given (PHY PER, code rate) operating point.

Paper 2 parameters (Table 1): SB=512 bytes, SS=64 bytes, K=SB/SS=8.
"""

import os
import numpy as np
import scipy.io


def load_raptor_q_table(mat_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Raptor-Q PER lookup table from the .mat file.

    MATLAB: load('K8_T512_Table_CR_PER_full.mat')
    Variables: mat_full_interp (PER table), CR axis = [1:-0.02:0.01]

    Args:
        mat_dir: Directory containing 'K8_T512_Table_CR_PER_full.mat'.

    Returns:
        (mat, per_axis, cr_axis)
        mat:      2-D PER table, shape (n_per, n_cr).
        per_axis: PHY PER axis values (rows), array of length n_per.
        cr_axis:  Code rate axis (columns), array of length n_cr.
    """
    fpath = os.path.join(mat_dir, "K8_T512_Table_CR_PER_full.mat")
    data = scipy.io.loadmat(fpath)
    mat = data["mat_full_interp"].astype(float)  # shape (n_per, n_cr)

    # Code rate axis (from MATLAB: CR = [1:-0.02:0.01])
    cr_axis = np.arange(1.0, 0.0, -0.02)   # [1.0, 0.98, ..., 0.02, 0.01... ]
    # Trim to table width
    n_cr = mat.shape[1]
    cr_axis = cr_axis[:n_cr]

    # PHY PER axis (rows): [1:-0.02:0.01]
    per_axis = np.arange(1.0, 0.0, -0.02)
    n_per = mat.shape[0]
    per_axis = per_axis[:n_per]

    return mat, per_axis, cr_axis


def raptor_q_prr_overhead(
    prr_phy: float,
    mat: np.ndarray,
    per_axis: np.ndarray,
    cr_axis: np.ndarray,
    target_per: float = 0.01,
    K: int = 8,
) -> tuple[float, float, float]:
    """
    Raptor-Q APP PRR and overhead via table lookup.

    Translated from Block 4 of main_modified_clean_LTE_final_Athirah.m:
        [x,idxR] = min(abs(1-sensor(k).prr_APP(kk,t)-per));
        idx = find(mat(idxR,:)<target);
        if ~isempty(idx)
            idxC = idx(1);  prr_rq = 1;
        else
            idxC = size(mat,2);  prr_rq = 1-mat(idxR,idxC);
        end
        overh_rq = K/CR(idxC);
        e2edelay_rq = overh_rq * e2edelay_APP;

    Args:
        prr_phy:    PHY-layer packet reception rate (not PER).
        mat:        PER table from load_raptor_q_table.
        per_axis:   PHY PER axis (rows).
        cr_axis:    Code rate axis (columns).
        target_per: Target PER (default 0.01).
        K:          Number of source symbols (default 8).

    Returns:
        (prr_rq, overhead, best_cr)
        prr_rq:   Raptor-Q PRR at optimal code rate.
        overhead: K/CR — encoding overhead (number of encoded symbols).
        best_cr:  Best achievable code rate.
    """
    per_phy = 1.0 - prr_phy  # convert PRR to PER for row lookup

    # Find closest row
    idx_r = int(np.argmin(np.abs(per_axis - per_phy)))

    # Find first code rate where table PER < target
    row = mat[idx_r, :]
    idx_below = np.where(row < target_per)[0]

    if len(idx_below) > 0:
        idx_c = int(idx_below[0])
        prr_rq = 1.0
    else:
        idx_c = mat.shape[1] - 1
        prr_rq = 1.0 - float(mat[idx_r, idx_c])

    best_cr  = float(cr_axis[idx_c])
    overhead = K / best_cr if best_cr > 0 else float("inf")

    return prr_rq, overhead, best_cr
