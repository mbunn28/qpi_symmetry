# Full T-matrix QPI (A1 impurity)

Replaces the Born approximation, T = V, by the full local T-matrix of `soc_materials/spin_qpi/qpi_fritz.py`:

    T(w) = [1 - V G_loc(w)]^-1 V,   V = v tau_z,   G_loc = (1/N^2) sum_k g0(k, w)

The method needs a point impurity (T independent of q), so only A1 is covered. Everything after T (S(q), drho_mu,
Gaussian cutoff, C6v analysis, plots) is the `2_new/qpi_simulation.py` pipeline, imported unchanged.

## Run

From this folder, in order (each step takes under a minute):

| Command | Produces |
|---|---|
| `python qpi_fullT.py` | `data/delta0.1_gamma0.001_alpha0.1/` (Rashba: s, ext s, d+id, dx2y2) and `data/delta0.1_gamma0.001_alpha0/` (no SOC: dx2y2, then s, ext s, d+id) |
| `python qpi_fullT.py nosoc` | only the no-SOC s, ext s, d+id runs, with no resonance search. They are used only by the case table (`../2026-11-06_nosoc_figs/case_table.py`) |
| `python qpi_fullT_frg.py` | `data/frg_<run>/delta_rms0.1_gamma0.001/` for the two divERGe runs |
| `python qpi_fullT.py nocut` | as `python qpi_fullT.py`, with no q cutoff, into `data_nocut/` (the note's full-T results) |
| `python qpi_fullT_frg.py nocut` | as `python qpi_fullT_frg.py`, with no q cutoff, into `data_nocut/` |
| `python checks.py` | `checks.txt`: validation, see below |
| `./note_assets.sh` | runs `resonant_table.py` (`resonant_table.tex`), then copies that table into `notes/2026-11-03_breaking_TRS/tables` (the full-T figures are made in `../2026-11-06_rashba_figs`) |

Smoke test, written outside `data/`: `python qpi_fullT.py smoke <dir>` (Nk = 48, also compares with `born_S`).

## Output per state

`<pairing>/<case>/` with the usual `symmetry_analysis_*.txt`, `drho_*.png`, `irreps_*.png`, `note_drho_*.pdf` and `drho_*.npz`
(the npz also stores `T` and `v`). The cases are:

- `born`: T = v tau_z, v = 0.1 (the reference)
- `full`: the full T at v = 0.1
- `res<n>`: the full T at each distinct resonant v

`tmatrix_<pairing>.txt` gives G_loc, the resonant v (or the complex lambda that were skipped), max|T| and min|eig(1 - V G_loc)| for each case.

## Decisions

- **No q cutoff for the note (`data_nocut/`, file tag `q0inf`).** T is the local T of a point impurity, so V(q) is constant and S(q) needs no Gaussian. Multiplying the full-T S(q) by exp(-q^2/q0^2), as in `data/`, is not the T-matrix of a Gaussian impurity, only a low-pass filter. It is A1, so irreps are unaffected, but amplitudes and ratios change. `born` in `data_nocut/` is then Born for a point impurity, not the q0 = |b1| Born of the note's Born figures. `data/` (q0 = |b1|) is kept because those figures and the case table's Born entries use its `born` outputs.

- **S(q) by FFT** (`fft_S`): the convolution theorem, i.e. the real-space product G(r) T G(-r) of qpi_fritz.py. It equals `born_S` to 1e-14.
- **Resonant v = 1/lambda**, with lambda the real eigenvalues of G_loc(w, Gamma = 0) tau_z, as in qpi_fritz.py. T itself uses the broadened G_loc, so it stays finite.
  - `eig`, not `eigh`: G_loc tau_z is not Hermitian when the local anomalous part is non-zero (s, ext s).
  - Complex lambda means no bound state; it is skipped and logged.
  - Repeated values are removed.
- **Resonances only for gapped states**, inside the gap: a bound state needs a gap. dx2y2 is nodal (`../2026-11-06_dx2y2_nodes`), so it runs with `resonances=False` at `omega_vals=[0, 0.15]`. 0.15 is about half the maximum gap on the Fermi surface.
- **Zero threshold** scales with max|T| instead of v.

## Checks (`checks.txt`)

- `fft_S` equals `born_S`.
- The `born` outputs equal the existing `2_new` runs (ansatz and FRG) and `2026-11-05_nosoc` (no-SOC dx2y2, w = 0), to at most 3e-12.

## Not reproducible from the current scripts

`data/delta0.1_gamma0.001_alpha0.1/dx2y2_old_meshgap/` was made by an earlier version that used w = 0.98 x (mesh gap) and resonances for dx2y2.
That gap is a mesh artefact, so these results are kept for reference only.
