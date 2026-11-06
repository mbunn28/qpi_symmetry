# Rashba figures for the breaking_TRS note

## Run

From this folder:

| Command | Produces |
|---|---|
| `python combined_bands.py` | `figs/bands_combined.{pdf,png}`; prints the BZ gap of each state |
| `python rashba_spin.py` | `figs/dpid_compare.pdf`, `figs/rashba_born_nonchiral.pdf` |
| `python split_T.py` | `split_T.txt`: the resonant (V0 = 1.33) drho split into the tau_0 and tau_z parts of T, max of each and overlap with Born; quoted in Sec. fullT |
| `./note_assets.sh` | runs both, then copies the three PDFs to `notes/2026-11-03_breaking_TRS/figs/` |

## bands_combined

- **Columns:** Rashba d+id ansatz (alpha = 0.1, mu = -0.1, |Delta| = 0.1) and the chiral FRG gap (`qpi_simulation_frg.RUN_DIR`, U = 8, chi = +1, Delta scale 0.1). The BdG Hamiltonians are built as in `2_new/qpi_simulation.py` and `qpi_simulation_frg.py`, with the same Delta as the QPI runs in `../2026-11-05_fullT`.
- **Rows:** normal; BdG with +-eps(k) dashed; BdG zoomed.
- **Zoom:** k-range around the normal-state crossings within 0.5 of the path's smallest |E|, padded by 1.5 x 4 gap / v_F; E-range +-4 gap, boxed in the BdG row. The shading is |E| < E_g, the gap on a 240^2 BZ mesh (0.2137 and 0.0863; these match omega_2 = 0.98 gap in the note).

## dpid_compare

- Fig. dpid_compare of the note: Rashba d+id, scalar impurity. Rows drho_0, x, y, z; column pairs (q, r):
  - Born at omega_2 = 0.2095, q0 = |b1| (`../2026-11-05_fullT/data/.../dpid/born/`);
  - full T at omega_2, V0 = 0.1, no cutoff (`data_nocut/.../dpid/full/`);
  - full T at omega_2 at the resonance V0 = 1.33, no cutoff (`data_nocut/.../dpid/res1/`; the sharper of the two omega_2 resonances, min|eig| 0.033 vs 0.062 at V0 = -2.28).
- **q panels:** Im for x, y and Re for 0, z (the other part is below 1e-10 of the max), scaled to the max over q != 0; every panel has its max below it (irreps are in the case table).

## rashba_born_nonchiral

- Fig. rashba_nonchiral: s, ext s, dx2y2, Born at omega_2 (0.98 gap; 0.15 for nodal dx2y2, see `../2026-11-06_dx2y2_nodes`), q0 = |b1|, drho_0 only; their spin channels are below `zero_tol`.
- Same panel conventions; the q = 0 peak is 3-6x the rest for ext s and dx2y2, so the centre saturates.
