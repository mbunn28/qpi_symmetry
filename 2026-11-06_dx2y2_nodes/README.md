# Is the dx2y2 ansatz state gapped?

No. It has point nodes, and its "gap" on the QPI k-mesh is a resolution artefact. The ext s gap is real.

## Run

`python nodes.py` writes `nodes.txt` (about 1 min).

## Method

Singlet pairing with Rashba SOC splits into helical bands, E_pm = sqrt(xi_pm^2 + |Psi(k)|^2) with xi_pm = eps +- alpha |grad eps|.
The script checks this against the full 4x4 BdG. The state is therefore gapless exactly when a zero line of Psi crosses a helical Fermi surface.

`nodes.py` computes:
1. min|E| on the QPI mesh for Nk = 60 to 1920;
2. the continuum minimum, found by zooming a grid around the lowest mesh points;
3. the Fermi surfaces, by radial bisection at 72000 angles, then the sign changes of Psi along them (the nodes) and the max |Delta| on them.

## Result (mu = -0.1, |Delta| = 0.1)

| | dx2y2, alpha = 0.1 | dx2y2, no SOC | ext s, alpha = 0.1 | ext s, no SOC |
|---|---|---|---|---|
| min\|E\| on the Nk = 240 mesh | 5.0e-3 | 1.07e-2 | 1.91e-2 | 1.00e-2 |
| continuum min\|E\| | 0 | 0 | 1.89e-2 | 9.95e-3 |
| nodes | 4 per helical FS (~49, 131, 229, 311 deg) | 4 | none | none |
| max\|Delta\| on the FS | 0.28 to 0.31 | 0.30 | | |

As a result, the dx2y2 QPI runs use a fixed w = 0.15 and no resonance search (`../2026-11-05_fullT`, `../2026-11-06_nosoc_figs`).
