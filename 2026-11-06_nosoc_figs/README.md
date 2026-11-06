# No-SOC combined figures and the spin-channel case table

## Run

From this folder:

| Command | Produces |
|---|---|
| `python combined_nosoc.py` | `figs/nosoc_A1imp_all.pdf`, `figs/nosoc_A2imp_all.pdf`, `figs/nosoc_A2imp_nonzero.pdf`; on the first run it also computes `data/A2_dx2y2_om0.1500_*.npz` (about 1 min) |
| `python case_table.py` | `case_table.tex` (the table rows, Born only) and `case_table_fullT.txt` (the full-T entries that differ from Born); the rows are also printed to the terminal |
| `./note_assets.sh` | runs `case_table.py`, then copies `case_table.tex` to `notes/2026-11-03_breaking_TRS/tables/cases.tex`, and `nosoc_A1imp_all.pdf` and `nosoc_A2imp_nonzero.pdf` to its `figs/` |

## Combined figures

- **Layout:** rows are s, ext s, d+id and dx2y2. Columns are Re drho(q) and drho(r) at w = 0, then the same at w_2.
- **w_2:** 0.98 x gap for the gapped states. For nodal dx2y2 it is 0.15 (`../2026-11-06_dx2y2_nodes`).
- **Style:** as `../2026-11-06_rashba_figs` dpid_compare: each panel is normalised to its own maximum, given below it; no irrep weights (they are in the case table), ticks or frames. "~0" means below the dataset's `zero_tol`.
- **nosoc_A1imp_all:** rows labelled (a)-(d), quoted in the case table, and separated by horizontal lines; column pairs headed w = 0 and w_2.
- **A2 in the note:** almost every A2 panel is zero, so the note uses `nosoc_A2imp_nonzero.pdf`. It has one row per (pairing, w) with a finite Re drho_0, which is only d+id at w_2. `nosoc_A2imp_all.pdf` is still made, for reference.
- **Why only Re drho_0:** without SOC, Im drho_0(q) and all spin channels are at round-off level.

**Data sources (Born approximation):**
- `../2026-11-05_nosoc/data/delta0.1_gamma0.001/{A1,A2}/`
- dx2y2 at w = 0.15:
  - A1 from `../2026-11-05_fullT/data/delta0.1_gamma0.001_alpha0/dx2y2/born/`. This is the same calculation as `qpi_nosoc`; see `../2026-11-05_fullT/checks.txt`.
  - A2 computed here with `qpi_nosoc.a2_S`, q0 = 20 |b1|.

## Case table

For each (system, pairing, impurity), the entry for each channel drho_mu(q), mu = 0, x, y, z, is the set of C6v irreps of its Re and Im parts with weight > 1e-3. It is "0" if both parts are below `zero_tol`, and "Im:" marks a purely imaginary channel.

- It is the union over w = 0 and w_2: dagger means finite only at w_2, double dagger only at w = 0.
- The table is Born only. Full T (A1 only, since it needs a point impurity) is compared in `case_table_fullT.txt`, from `../2026-11-05_fullT/data/*/<pairing>/full/`. The no-SOC full-T runs come from `python qpi_fullT.py nosoc`. The only difference is Rashba d+id, whose spin channels are also finite at w = 0; the note states this in the caption and in Sec. fullT instead of adding rows.
- Pairings are labelled by their own irrep, hard-coded from `../2026-11-06_pairing_irreps/pairing_irreps.txt`.
- FRG labels give U and V from `../2026-11-06_pairing_irreps/frg_params.txt`.
- The last column lists the note's figures that plot each case's QPI (`FIGS` in the script, by figure label); "--" means none.
- Rashba with the A2 impurity has not been computed.
