# QPI maps with the q = 0 peak suppressed

The spin-resolved Born QPI of Figs 4 (Rashba, d+id, omega = 0.2095) and 6 (FRG U = 8, chi = +1, omega = 0.0846) of
`notes/2026-11-03_breaking_TRS`, with the q = 0 peak suppressed as in experiment: by saturating the colour scale (cutoff) or blanking a disc around q = 0 (mask).

## Run

`python qpi_cutoff.py` (a few seconds) writes `figs/cutoff_<case>.pdf` and `figs/mask_<case>.pdf` for `<case>` = `rashba_dpid`, `frg_chiral`.
It only reads the saved Born npz files in `../2026-11-05_fullT/data` (the same data as the note figures); nothing is recomputed.

## Method

**Cutoff.** For each q-space panel, with M = max|drho(q)|, and cutoff c in `CUTOFFS` (1, 0.9, 0.75, 0.5, 0.25):

    shown = clip(drho(q), -c M, c M) / (c M)

so values above c M saturate at +-1. The corner number is c M.
- Rows: the finite q-space panels of the note figure: Re drho_0, Im drho_x, Im drho_y, Re drho_z.
- Columns: the cutoffs; "no cutoff" reproduces the note figure.
- The r-space panels are left out, because the cutoff targets the q = 0 peak.

**Mask.** For each q-space panel and mask radius r in `MASKS` (0, 0.02, 0.05, 0.1 |b1|; mesh spacing |b1|/120):

    shown = drho(q) / max_{|q| >= r} |drho(q)|,   blank (NaN) for |q| < r

|q| is folded to the first BZ. Nothing saturates; the corner number is the max outside the disc. Same rows as above.
