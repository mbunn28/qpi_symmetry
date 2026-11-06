# Rashba, scalar impurity. Four note figures:
#   dpid_compare.pdf: d+id at omega_2, rows drho_0, x, y, z; column pairs (q, r) for Born (q0 = |b1|),
#                     full T at V0 = 0.1 and full T at the resonance V0 = 1.33 (no q cutoff).
#   rashba_born_nonchiral.pdf: s, ext s, dx2y2, Born at omega_2, drho_0 only (their spin channels vanish).
#   frg_compare.pdf: chiral FRG state at its gap edge, as dpid_compare (resonance V0 = 1.22).
#   frg_nonchiral.pdf: the U = 0 FRG state, as frg_compare; drho_0 only (its spin channels vanish).
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
FRG_RUN = os.path.join('frg_nu0.5_U8_a0.4_nk24_nkf31', 'delta_rms0.1_gamma0.001', 'frg_chi+1')
FRG, FRG_NOCUT = os.path.join(FULLT, 'data', FRG_RUN), os.path.join(FULLT, 'data_nocut', FRG_RUN)
FRG0_RUN = os.path.join('frg_nu0.5_U0_a0.1_V0.9_nk36_nkf25_c4.01', 'delta_rms0.1_gamma0.001', 'frg')
FRG0, FRG0_NOCUT = os.path.join(FULLT, 'data', FRG0_RUN), os.path.join(FULLT, 'data_nocut', FRG0_RUN)
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


def compare(P, cases, path, figsize, channels=range(4), cbar_h=0.12):
    """Column pair (q, r) per case, one row per channel (non-zero ones only); vertical dividers between cases.
    A single channel goes in the column titles instead of a row label, as rashba_born_nonchiral."""
    single = len(channels) == 1
    sym = CH[channels[0]] if single else 'i'
    fig = plt.figure(figsize=figsize, layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, cbar_h])
    wr = [1 if single else 1.1] + [1]*(len(cases) - 1)
    subs = top.subfigures(1, len(cases), wspace=0.03, width_ratios=wr)
    for k, (sf, (title, d)) in enumerate(zip(subs, cases)):
        sf.suptitle(title, fontsize=11)
        sf.get_layout_engine().set(h_pad=0.01, hspace=0)
        ax = sf.subplots(len(channels), 2, squeeze=False)
        for i, m in enumerate(channels):
            P.pair(ax[i, 0], ax[i, 1], d, m)
            if k == 0 and not single:
                ax[i, 0].set_ylabel(f'${CH[m]}$', rotation=0, labelpad=10, va='center', fontsize=15)
        ax[0, 0].set_title(rf'$\delta\rho_{sym}(\bvec{{q}})$', fontsize=12)
        ax[0, 1].set_title(rf'$\delta\rho_{sym}(\bvec{{r}})$', fontsize=12)
    for x in np.cumsum(wr)[:-1] / sum(wr):  # dividers between the cases
        top.add_artist(Line2D([x, x], [0.01, 0.99], transform=top.transSubfigure, color='0.5', lw=0.8))
    bottom.colorbar(ScalarMappable(NORM, CMAP), cax=bottom.add_axes([0.35, 0.7, 0.3, 0.12]),
                    orientation='horizontal', ticks=[-1, 0, 1], label=r'value / $\max|\cdot|$')
    fig.savefig(path, dpi=300)
    plt.close(fig)


def dpid_compare(P, path):
    cases = [(r'Born, $V_0 = 0.1$', omega2('dpid')),
             (r'Full $T$, $V_0 = 0.1$', np.load(os.path.join(NOCUT, 'full', 'drho_om0.2095_V0.1000_q0inf.npz'))),
             (r'Full $T$, $V_0 = 1.33$', np.load(os.path.join(NOCUT, 'res1', 'drho_om0.2095_V1.3279_q0inf.npz')))]
    compare(P, cases, path, (7.0, 5.0))


def frg_compare(P, path):
    """Chiral FRG state (U = 8, alpha = 0.4, chi = +1) at the gap edge: Born (q0 = |b1|), full T at V0 = 0.1 and at
    the sharpest gap-edge resonance V0 = 1.22 (res3; no cutoff)."""
    cases = [(r'Born, $V_0 = 0.1$', np.load(os.path.join(FRG, 'born', 'drho_om0.0846_V0.1000_q01.0000.npz'))),
             (r'Full $T$, $V_0 = 0.1$', np.load(os.path.join(FRG_NOCUT, 'full', 'drho_om0.0846_V0.1000_q0inf.npz'))),
             (r'Full $T$, $V_0 = 1.22$', np.load(os.path.join(FRG_NOCUT, 'res3', 'drho_om0.0846_V1.2163_q0inf.npz')))]
    compare(P, cases, path, (7.0, 5.0))


def frg_nonchiral(P, path):
    """Non-chiral FRG state (U = 0, alpha = 0.1) at omega = 0.0761, as frg_compare; only drho_0 is finite."""
    cases = [(r'Born, $V_0 = 0.1$', np.load(os.path.join(FRG0, 'born', 'drho_om0.0761_V0.1000_q01.0000.npz'))),
             (r'Full $T$, $V_0 = 0.1$', np.load(os.path.join(FRG0_NOCUT, 'full', 'drho_om0.0761_V0.1000_q0inf.npz')))]
    compare(P, cases, path, (4.9, 2.1), channels=[0], cbar_h=0.35)


def nonchiral(P, path):
    """Rows (a)-(c) separated by horizontal lines, as nosoc_A1imp_all; bottom colorbar."""
    fig = plt.figure(figsize=(3.4, 5.2), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.15])
    top.get_layout_engine().set(h_pad=0.01, hspace=0)
    ax = top.subplots(3, 2)
    for i, (folder, tex) in enumerate(NONCHIRAL):
        d = omega2(folder)
        P.pair(ax[i, 0], ax[i, 1], d, 0)
        ax[i, 0].set_ylabel(tex + '\n' + rf'$\omega_2 = {float(d["omega"]):.4f}$', fontsize=11)
        ax[i, 0].text(-0.42, 1.0, f'({"abc"[i]})', transform=ax[i, 0].transAxes, ha='right', va='top', fontsize=12)  # quoted in the case table
    ax[0, 0].set_title(r'$\delta\rho_0(\bvec{q})$', fontsize=12)
    ax[0, 1].set_title(r'$\delta\rho_0(\bvec{r})$', fontsize=12)
    bottom.colorbar(ScalarMappable(NORM, CMAP), cax=bottom.add_axes([0.25, 0.7, 0.5, 0.12]),
                    orientation='horizontal', ticks=[-1, 0, 1], label=r'value / $\max|\cdot|$')
    fig.canvas.draw()  # fix the layout, then place the dividers between rows
    inv = top.transSubfigure.inverted()
    for i in range(2):
        y = (inv.transform(ax[i, 0].get_tightbbox().p0)[1] + inv.transform(ax[i + 1, 0].get_tightbbox().p1)[1]) / 2
        top.add_artist(Line2D([0.01, 0.99], [y, y], transform=top.transSubfigure, color='0.5', lw=0.8))
    fig.set_layout_engine('none')  # keep the measured layout on save
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    P = Panels()
    dpid_compare(P, os.path.join(out, 'dpid_compare.pdf'))
    nonchiral(P, os.path.join(out, 'rashba_born_nonchiral.pdf'))
    frg_compare(P, os.path.join(out, 'frg_compare.pdf'))
    frg_nonchiral(P, os.path.join(out, 'frg_nonchiral.pdf'))
