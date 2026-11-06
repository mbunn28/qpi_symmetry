# Figs 4 and 6 of breaking_TRS.tex (spin-resolved Born QPI), with the q = 0 peak suppressed in two ways:
# cutoff: |drho(q)| above c*max is clipped to c*max, then scaled by c*max;
# mask: drho(q) blanked for |q| < r|b1|, then scaled by the max outside. Saved data only; nothing is recomputed.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
from qpi_simulation import setup_mesh, fold_to_ws, draw_panel, corner_note, add_colorbar  # noqa: E402

FULLT = os.path.join(ROOT, '2026-11-05_fullT', 'data')
# (output name, Born npz used by the note figure)
CASES = [('rashba_dpid', os.path.join(FULLT, 'delta0.1_gamma0.001_alpha0.1', 'dpid', 'born', 'drho_om0.2095_V0.1000_q01.0000.npz')),
         ('frg_chiral', os.path.join(FULLT, 'frg_nu0.5_U8_a0.4_nk24_nkf31', 'delta_rms0.1_gamma0.001', 'frg_chi+1', 'born',
                                     'drho_om0.0846_V0.1000_q01.0000.npz'))]
CUTOFFS = [1.0, 0.9, 0.75, 0.5, 0.25]  # fraction of max|drho(q)| per panel; 1 = no cutoff (as in the note)
MASKS = [0.0, 0.02, 0.05, 0.1]  # mask radius / |b1|; the q-mesh spacing is |b1|/120

lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


def finite_rows(d):
    """Finite q-space panels (Re/Im per channel), as in the note: [(label, img)]."""
    rows = []
    for m, ch in enumerate(['0', 'x', 'y', 'z']):
        sub = f'_{ch}' if ch != '0' else ''
        for part, img in [('Re', np.real(d['drho'][m])), ('Im', np.imag(d['drho'][m]))]:
            if np.amax(np.abs(img)) >= d['zero_tol']:
                rows.append((rf'$\mathrm{{{part}}}\,\delta\rho{sub}(\bvec{{q}})$', img))
    return rows


def plot(npz, path, Nq=120):
    """Rows: finite q-space panels; columns: cutoffs."""
    qmesh = setup_mesh(Nq, rlattice)
    rows = finite_rows(np.load(npz))
    fig, ax = plt.subplots(len(rows), len(CUTOFFS), figsize=(1.35*len(CUTOFFS) + 0.8, 1.4*len(rows)),
                           layout='constrained', squeeze=False)
    for i, (label, img) in enumerate(rows):
        vmax = np.amax(np.abs(img))
        for j, c in enumerate(CUTOFFS):
            a = ax[i, j]
            draw_panel(a, 'q', np.clip(img, -c*vmax, c*vmax) / (c*vmax), qmesh, rlattice, None)
            corner_note(a, f'{c*vmax:.2g}')
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title('no cutoff' if c == 1 else rf'{100*c:.0f}\% of max')
        ax[i, 0].set_ylabel(label)
    add_colorbar(fig, ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_mask(npz, path, Nq=120):
    """Rows: finite q-space panels; columns: mask radii. Masked pixels are left blank (NaN)."""
    qmesh = setup_mesh(Nq, rlattice)
    qabs = np.linalg.norm(fold_to_ws(qmesh, rlattice), axis=-1) / np.linalg.norm(rlattice[0])
    rows = finite_rows(np.load(npz))
    fig, ax = plt.subplots(len(rows), len(MASKS), figsize=(1.35*len(MASKS) + 0.8, 1.4*len(rows)),
                           layout='constrained', squeeze=False)
    for i, (label, img) in enumerate(rows):
        for j, r in enumerate(MASKS):
            a = ax[i, j]
            z = np.where(qabs < r, np.nan, img)
            vmax = np.nanmax(np.abs(z))
            draw_panel(a, 'q', z / vmax, qmesh, rlattice, None)
            corner_note(a, f'{vmax:.2g}')
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title('no mask' if r == 0 else rf'$|\bvec{{q}}| < {r:g}\,|\bvec{{b}}_1|$')
        ax[i, 0].set_ylabel(label)
    add_colorbar(fig, ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    for name, npz in CASES:
        plot(npz, os.path.join(out, f'cutoff_{name}.pdf'))
        plot_mask(npz, os.path.join(out, f'mask_{name}.pdf'))
