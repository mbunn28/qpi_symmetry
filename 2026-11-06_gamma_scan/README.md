# Broadening (Gamma) scan, Rashba d+id

## Run

| Command | Produces |
|---|---|
| `python gamma_scan.py > gamma_scan.txt` | max over q of each \|drho_mu(q)\| (mu = 0, x, y, z) and z/0, Born and full T at V0 = 0.1, for Gamma = 2e-3, 1e-3, 5e-4, 2.5e-4 at omega = 0 and omega_2 = 0.98 gap; t0 is the tau_0 part of T |

Same model and pipeline as `../2026-11-05_fullT` (mu = -0.1, alpha = 0.1, |Delta| = 0.1, Nk = 240, Nq = 120, no q cutoff). Takes a few minutes.

## Result

- Both frequencies are inside the gap, so every channel scales roughly as Gamma; the ratios do not depend on Gamma (z/0 = 6.2e-4 at omega = 0 and 0.27 at omega_2, full T).
- At omega = 0, the Born spin channels are zero (~1e-14); the full-T ones come only from the tau_0 part of T, t0 = -i V0^2 eta / D with eta = -Im G_loc(0).
