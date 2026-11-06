# Rashba, scalar impurity. Two note figures:
#   dpid_compare.pdf: d+id at omega_2, rows drho_0, x, y, z; column pairs (q, r) for Born (q0 = |b1|),
#                     full T at V0 = 0.1 and full T at the resonance V0 = 1.33 (no q cutoff).
#   rashba_born_nonchiral.pdf: s, ext s, dx2y2, Born at omega_2, drho_0 only (their spin channels vanish).
# Data from ../2026-11-05_fullT.
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
from matplotlib.lines import Line2D  # noqa: E402
from qpi_simulation import setup_mesh, draw_panel, CMAP, NORM  # noqa: E402

FULLT = os.path.join(ROOT, '2026-11-05_fullT')
DATA = os.path.join(FULLT, 'data', 'delta0.1_gamma0.001_alpha0.1')
NOCUT = os.path.join(FULLT, 'data_nocut', 'delta0.1_gamma0.001_alpha0.1', 'dpid')
NONCHIRAL = [('s', '$s$'), ('ext_s', r'ext.\ $s$'), ('dx2y2', '$d_{x^2-y^2}$')]
CH = ['0', 'x', 'y', 'z']
IM = (False, True, True, False)  # x, y are imaginary in q, 0 and z real
lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


def omega2(folder):
    fs = glob.glob(os.path.join(DATA, folder, 'born', 'drho_*.npz'))
    return np.load(max(fs, key=lambda f: float(np.load(f)['omega'])))



class Panels:
    def __init__(self, Nq=120, r_window=10):
        self.qmesh = setup_mesh(Nq, rlattice)
        self.geom = dict(q=(self.qmesh, rlattice), r=(setup_mesh(Nq, lattice) * Nq, lattice))
        self.gamma = np.all(np.abs(self.qmesh) < 1e-9, axis=-1)
        self.r_window = r_window

    def pair(self, axq, axr, d, m):
        """Re (0, z) or Im (x, y) drho_m(q) and drho_m(r), each with its max below; q scaled to max over q != 0."""
        img_q = np.imag(d['drho'][m]) if IM[m] else np.real(d['drho'][m])
        for a, kind, img in [(axq, 'q', img_q), (axr, 'r', np.real(d['drho_r'][m]))]:
            vmax = np.amax(np.abs(np.where(self.gamma, 0, img) if kind == 'q' else img))
            draw_panel(a, kind, img/vmax, *self.geom[kind], self.r_window)
            a.set_xlabel(f'{vmax:.2g}', labelpad=1, fontsize=8)
            a.set_xticks([])
            a.set_yticks([])
            a.spines[:].set_visible(False)


def colorbar(fig, ax):
    fig.colorbar(ScalarMappable(NORM, CMAP), ax=ax, orientation='horizontal', shrink=0.4, aspect=30,
                 ticks=[-1, 0, 1], label=r'value / $\max|\cdot|$')


def dpid_compare(P, path):
    born = omega2('dpid')
    cases = [(r'Born, $V_0 = 0.1$', born),
             (r'Full $T$, $V_0 = 0.1$', np.load(os.path.join(NOCUT, 'full', 'drho_om0.2095_V0.1000_q0inf.npz'))),
             (r'Full $T$, $V_0 = 1.33$', np.load(os.path.join(NOCUT, 'res1', 'drho_om0.2095_V1.3279_q0inf.npz')))]
    fig = plt.figure(figsize=(7.0, 5.0), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.12])
    wr = [1.1, 1, 1]
    subs = top.subfigures(1, 3, wspace=0.03, width_ratios=wr)
    for k, (sf, (title, d)) in enumerate(zip(subs, cases)):
        sf.suptitle(title, fontsize=11)
        sf.get_layout_engine().set(h_pad=0.01, hspace=0)
        ax = sf.subplots(4, 2)
        for m in range(4):
            P.pair(ax[m, 0], ax[m, 1], d, m)
            if k == 0:
                ax[m, 0].set_ylabel(f'${CH[m]}$', rotation=0, labelpad=10, va='center', fontsize=15)
        ax[0, 0].set_title(r'$\delta\rho_i(\bvec{q})$', fontsize=12)
        ax[0, 1].set_title(r'$\delta\rho_i(\bvec{r})$', fontsize=12)
    for x in np.cumsum(wr)[:-1] / sum(wr):  # dividers between the cases
        top.add_artist(Line2D([x, x], [0.01, 0.99], transform=top.transSubfigure, color='0.5', lw=0.8))
    bottom.colorbar(ScalarMappable(NORM, CMAP), cax=bottom.add_axes([0.35, 0.7, 0.3, 0.12]),
                    orientation='horizontal', ticks=[-1, 0, 1], label=r'value / $\max|\cdot|$')
    fig.savefig(path, dpi=300)
    plt.close(fig)


def nonchiral(P, path):
    fig, ax = plt.subplots(3, 2, figsize=(3.4, 4.6), layout='constrained')
    for i, (folder, tex) in enumerate(NONCHIRAL):
        d = omega2(folder)
        P.pair(ax[i, 0], ax[i, 1], d, 0)
        ax[i, 0].set_ylabel(tex + '\n' + rf'$\omega_2 = {float(d["omega"]):.4f}$', fontsize=11)
    ax[0, 0].set_title(r'$\delta\rho_0(\bvec{q})$', fontsize=12)
    ax[0, 1].set_title(r'$\delta\rho_0(\bvec{r})$', fontsize=12)
    colorbar(fig, ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    P = Panels()
    dpid_compare(P, os.path.join(out, 'dpid_compare.pdf'))
    nonchiral(P, os.path.join(out, 'rashba_born_nonchiral.pdf'))
