# Rashba figures for the breaking_TRS note

## Run

From this folder:

| Command | Produces |
|---|---|
| `python combined_bands.py` | `figs/bands_combined.{pdf,png}`; prints the BZ gap of each state |
| `python rashba_spin.py` | `figs/rashba_spin_all.pdf` |
| `./note_assets.sh` | runs both, then copies the two PDFs to `notes/2026-11-03_breaking_TRS/figs/` |

## bands_combined

- **Columns:** Rashba d+id ansatz (alpha = 0.1, mu = -0.1, |Delta| = 0.1) and the chiral FRG gap (`qpi_simulation_frg.RUN_DIR`, U = 8, chi = +1, Delta scale 0.1). The BdG Hamiltonians are built as in `2_new/qpi_simulation.py` and `qpi_simulation_frg.py`, with the same Delta as the QPI runs in `../2026-11-05_fullT`.
- **Rows:** normal; BdG with +-eps(k) dashed; BdG zoomed.
- **Zoom:** k-range around the normal-state crossings within 0.5 of the path's smallest |E|, padded by 1.5 x 4 gap / v_F; E-range +-4 gap, boxed in the BdG row. The shading is |E| < E_g, the gap on a 240^2 BZ mesh (0.2137 and 0.0863; these match omega_2 = 0.98 gap in the note).

## rashba_spin_all

- Fig. rashba_spin of the note: Rashba, scalar impurity, Born, one frequency per state, omega_2 (0.98 gap; 0.15 for nodal dx2y2, see `../2026-11-06_dx2y2_nodes`). Only non-zero panels.
- **Left:** d+id, rows drho_0, x, y, z, each as (q, r). **Right:** s, ext s, dx2y2, drho_0 only; their spin channels are below `zero_tol`.
- **Data:** Born runs in `../2026-11-05_fullT/data/delta0.1_gamma0.001_alpha0.1/<pairing>/born/`, v = 0.1, q0 = |b1|.
- **q panels:** the larger of Re and Im, with C6v weights below, scaled to the max over q != 0 (given in the corner); the q = 0 peak is 3-6x the rest for ext s and dx2y2, so the centre saturates.
