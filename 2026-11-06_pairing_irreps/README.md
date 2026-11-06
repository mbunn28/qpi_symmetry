# C6v irrep of each pairing state

Gives the irrep labels of the pairings in the case table (Table I of `notes/2026-11-03_breaking_TRS`).

## Run

- `python pairing_irreps.py` writes `pairing_irreps.txt` (a few seconds; needs `diverge` and the FRG runs in `StAndrews/triangular_rashba`).
- `python frg_params.py` writes `frg_params.txt`. It lists U, V, J and mu from each FRG run's `.input` (`absent` means the key is not set, i.e. 0), plus the filling nu: the fraction of normal-state states below mu, computed on a 600^2 mesh from the hr.dat and mu that the QPI code uses. Both runs are at nu = 0.5 (half filling); U = 0 has V = 0.9 and alpha = 0.1, U = 8 has V = 0 and alpha = 0.4.

## Method

Delta(k) is the 2x2 spin gap matrix: Psi(k) i sigma_y for the ansatz states, and the divERGe vector (the chiral combination (v0 + i v1)/sqrt2 for U = 8) for the FRG states.
It is projected as in `2_new`, Delta_nu = (d_nu/|G|) sum_g chi_nu(g) g.Delta, with spin acting as well:

    (g.Delta)(k) = D(g) Delta(R_g^-1 k) D(g)^T,   D(g) in SU(2)   (the sign of D cancels)

The script scans the conventions: rotation sense, mirror spin axis, and D vs D*.
- Singlets are insensitive to all of them, since D i sigma_y D^T = i sigma_y.
- Two conventions, (-1, 2, True) and (1, 2, False), give pure irreps for both FRG states. They are the same operation: D* for Delta, because the BdG stores the conjugate gap (`2_new/qpi_simulation_frg.py`, `calculate_bdg`).
- The other conventions mix irreps (e.g. 6% A1 + 6% A2 for U = 8), which a single FRG eigenvector cannot do, so they are wrong.

## Result

| State | singlet fraction | irrep |
|---|---|---|
| s, ext s | 1 | A1 |
| d+id, dx2y2 | 1 | E2 |
| FRG U = 0 (alpha 0.1) | 0.997 | A1 |
| FRG U = 8 (alpha 0.4), chi = +1 | 0.755 | E2 |
