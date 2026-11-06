# QPI symmetry project — context for a new session (written 2026-11-05)

## Goal
Show how quasiparticle interference (QPI) from a single impurity in a superconductor on the triangular lattice reveals the symmetry of the pairing, in particular time-reversal breaking (chiral d+id). Each spin-resolved FT-LDOS map is decomposed into C6v irreps.

## Layout
- `1_basic/`: the first, spin-degenerate scripts (A1 and A2 impurities). These are legacy, kept for reference.
- `2_new/qpi_simulation.py`: the current machinery. It covers the Rashba model with ansatz pairings (s, ext s, d+id, dx2y2).
- `2_new/qpi_simulation_frg.py`: the same pipeline with the pairing read from divERGe FRG runs. Those runs live in `~/Library/CloudStorage/OneDrive-Personal/Documents/StAndrews/triangular_rashba/`.
- `2026-11-05_nosoc/qpi_nosoc.py`: the no-SOC rerun of `1_basic`. It imports its functions from `2_new/qpi_simulation.py` through `sys.path`. Use it as a template for a new case.
- `notes/2026-11-03_breaking_TRS/breaking_TRS.tex`: the working note (revtex4-2, built with latexmk). Its "Sn/Si(111) Wannier model" results section is still empty.
- `notes/QPI.pdf`: the conventions for S(q) and the FT-LDOS.

## Model and conventions
- **Basis:** spinful Nambu (c_k↑, c_k↓, c†_-k↑, c†_-k↓). The BdG Hamiltonian is H_BdG = [[h(k), Δ(k)], [Δ†, −h^T(−k)]].
- **Normal state:** h = ε(k) + α(∂_ky ε σ_x − ∂_kx ε σ_y), with μ = −0.1 and α = 0.1. Set α = 0 for the no-SOC case.
- **Pairing:** singlet, Δ = Ψ(k) iσ_y, scaled by `delta_mag`.
- **Born approximation:** S(q) = Σ_k g0(k) T g0(k−q).
- **Spin-resolved LDOS:** δρ_μ(q) = −[Tr σ_μ S(q) − Tr σ_μ S(−q)†]/(2πi), using the particle block only and μ ∈ {0, x, y, z}.
- **Gaussian cutoff:** e^{−q²/q0²}, with q0 = q0_factor·|b1|.
- **Impurities:**
  - A1: T = v τ_z.
  - A2: V(k,k') = i v (k×k')_z τ_z, with k folded to the Wigner–Seitz cell.
  - Keep τ_z here. The user said not to change the Nambu structure.
- **Symmetry analysis:** C6v irrep weights Σ|f_ν|²/Σ|f|². A dataset with max|f| below `zero_tol` counts as numerically zero.
- **Current parameters:** |Δ| = 0.1, Γ = 1e-3, Nk = 240, Nq = 120 (Nk must be a multiple of Nq), v = 0.1.
- **Energies:** ω ∈ {0, 0.98·gap}.

## Outputs per case
For each case the scripts write:
- `symmetry_analysis_*.txt`
- `drho_*.png` (all panels, with title)
- `irreps_*.png`
- `note_drho_*.pdf` (from `plot_note`: no title, non-zero panels only; this is the version used in the note)
- `drho_*.npz`
- `bands_*.png`

## Key physics so far
- **No SOC:** all spin channels vanish. With the A1 impurity, d+id gives only A1. The A2 impurity response is non-zero only for d+id near the gap edge. dx2y2 shows an E2 (nematic) component.
- **Rashba:** d+id gives δρ_z ∈ A1 and Im δρ_{x,y} ∈ E1. This is the signature of broken time reversal plus broken mirror symmetry.
- **FRG:** U=8, α=0.4 is degenerate, so the chiral combination (v0 + iχ v1)/√2 is used. U=0, α=0.1 is non-degenerate.
- **Resolution caveat:** with |Δ| = 0.1, the ext s and dx2y2 gaps (~0.007) are below the k-mesh level spacing. Treat those gap-edge results with care.

## Rules for this session
- **Don't edit `2_new/qpi_simulation.py` or `2_new/qpi_simulation_frg.py`** while the other session's queued runs are pending. Put the new case in its own dated folder (`2026-11-XX_<case>/`) and import from `2_new`, as `qpi_nosoc.py` does.
- **Leave `breaking_TRS.tex` alone:** the other session is updating it with the new figures.
- **Watch CPU load:** heavy runs will slow both sessions, so smoke-test at a small Nk first. A smoke test must not write into existing data folders.
- **Working style:**
  - Stay in scope.
  - Keep comments terse.
  - Don't rerun when a change can't alter the results.
  - Subagents run on Sonnet with narrow briefs.
