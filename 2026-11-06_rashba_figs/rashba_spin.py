# Rashba, scalar impurity, Born, at omega_2 (0.98 gap; 0.15 for nodal dx2y2): only the non-zero panels.
# Left: d+id, rows drho_0, x, y, z (q and r). Right: s, ext s, dx2y2, drho_0 only (their spin channels vanish).
# Data from ../2026-11-05_fullT (born = qpi_simulation's Born S).
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
from qpi_simulation import (setup_mesh, build_c6v_perms, irrep_decomposition, draw_panel,  # noqa: E402
                            corner_note, IRREP_NAMES, CMAP, NORM)

DATA = os.path.join(ROOT, '2026-11-05_fullT', 'data', 'delta0.1_gamma0.001_alpha0.1')
NONCHIRAL = [('s', '$s$'), ('ext_s', r'ext.\ $s$'), ('dx2y2', '$d_{x^2-y^2}$')]
CH_TEX = [r'\delta\rho', r'\delta\rho_x', r'\delta\rho_y', r'\delta\rho_z']
lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


def omega2(folder):
    fs = glob.glob(os.path.join(DATA, folder, 'born', 'drho_*.npz'))
    return np.load(max(fs, key=lambda f: float(np.load(f)['omega'])))


def weights_label(img, mesh, U, min_weight=1e-3):
    _, w, _ = irrep_decomposition(img, mesh, U)
    return ', '.join(rf'{IRREP_NAMES[n][0]}$_{IRREP_NAMES[n][1]}$ {100*w[n]:.0f}\%' for n in np.argsort(-w) if w[n] > min_weight)


def main(path, Nq=120, r_window=10):
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq = build_c6v_perms(qmesh)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))
    gamma = np.all(np.abs(qmesh) < 1e-9, axis=-1)

    def pair(axq, axr, d, m):
        """Re or Im (the larger) drho_m(q) and drho_m(r); q scaled to max over q != 0, where the q = 0 peak dominates."""
        re, im = np.real(d['drho'][m]), np.imag(d['drho'][m])
        part, img_q = ('Re', re) if np.amax(np.abs(re)) >= np.amax(np.abs(im)) else ('Im', im)
        for a, kind, img, lab in [(axq, 'q', img_q, rf'$\mathrm{{{part}}}\,{CH_TEX[m]}(\bvec{{q}})$'),
                                  (axr, 'r', np.real(d['drho_r'][m]), rf'${CH_TEX[m]}(\bvec{{r}})$')]:
            vmax = np.amax(np.abs(np.where(gamma, 0, img) if kind == 'q' else img))
            draw_panel(a, kind, img/vmax, *geom[kind], r_window)
            a.set_title(lab)
            corner_note(a, f'{vmax:.2g}')
            if kind == 'q':
                a.set_xlabel(weights_label(img, qmesh, Uq), labelpad=1)
            a.set_xticks([])
            a.set_yticks([])

    fig = plt.figure(figsize=(7.0, 6.6), layout='constrained')
    left, right = fig.subfigures(1, 2, wspace=0.03)
    d = omega2('dpid')
    left.suptitle(rf'$d+id$, $\omega_2 = {float(d["omega"]):.4f}$')
    ax = left.subplots(4, 2)
    for m in range(4):
        pair(ax[m, 0], ax[m, 1], d, m)
    right.suptitle(r'Non-chiral states: spin channels $\approx 0$')
    ax = right.subplots(4, 2)
    for i, (folder, tex) in enumerate(NONCHIRAL):
        d = omega2(folder)
        pair(ax[i, 0], ax[i, 1], d, 0)
        ax[i, 0].set_ylabel(tex + rf', $\omega_2 = {float(d["omega"]):.4f}$')
    for a in ax[3]:
        a.set_axis_off()
    right.colorbar(ScalarMappable(NORM, CMAP), ax=ax[3, :], orientation='horizontal', shrink=0.6, ticks=[-1, 0, 1],
                   label=r'value / $\max|\cdot|$')
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    main(os.path.join(out, 'rashba_spin_all.pdf'))
