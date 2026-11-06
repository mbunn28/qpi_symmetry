# Combined no-SOC figures, one per impurity: rows = pairings, columns = (Re drho(q), drho(r)) at omega = 0 and omega_2.
# omega_2 = 0.98 gap for gapped states, 0.15 for nodal dx2y2. Only Re drho_0 is non-zero without SOC.
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2026-11-05_nosoc'))
from qpi_nosoc import a2_S  # noqa: E402  (also puts 2_new on sys.path)
from qpi_simulation import (PAULI, setup_mesh, neg_index, fold_to_ws, calculate_bloch,  # noqa: E402
                            calculate_sc_order_params, calculate_bdg, calculate_greens_function,
                            draw_panel, CMAP, NORM)

NOSOC = os.path.join(ROOT, '2026-11-05_nosoc', 'data', 'delta0.1_gamma0.001')
FULLT = os.path.join(ROOT, '2026-11-05_fullT', 'data', 'delta0.1_gamma0.001_alpha0', 'dx2y2', 'born')
OMEGA_NODAL = 0.15
PAIRINGS = [('s', 's', '$s$'), ('ext_s', 'ext s', r'ext.\ $s$'), ('dpid', 'd+id', '$d+id$'), ('dx2y2', 'dx2y2', '$d_{x^2-y^2}$')]

lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


def a2_dx2y2_nodal(v_mag=0.1, q0_factor=20.0, mu=-0.1, delta_mag=0.1, gamma=1e-3, Nk=240, Nq=120):
    """A2 impurity, dx2y2, omega = OMEGA_NODAL, Born; as qpi_nosoc.main. Cached in data/."""
    path = os.path.join(HERE, 'data', f'A2_dx2y2_om{OMEGA_NODAL:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.npz')
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kmesh, qmesh = setup_mesh(Nk, rlattice), setup_mesh(Nq, rlattice)
    kws = fold_to_ws(kmesh, rlattice)
    h = calculate_bloch(mu, 0.0, kmesh, lattice)
    en, u = np.linalg.eigh(calculate_bdg(h, calculate_bloch(mu, 0.0, -kmesh, lattice),
                                         delta_mag*calculate_sc_order_params(kmesh, lattice, 'dx2y2')))
    g0 = calculate_greens_function(np.array([OMEGA_NODAL]), gamma, en, u)[0]
    S = a2_S(g0, kws, v_mag, Nq, Nk // Nq)
    q0 = np.linalg.norm(rlattice[0])*q0_factor
    S = S * np.exp(-np.sum(fold_to_ws(qmesh, rlattice)**2, axis=-1) / q0**2)[:, :, None, None]
    neg = neg_index(Nq)
    tr = np.einsum('mab,ijba->mij', PAULI, S[..., :2, :2])  # all four channels, as qpi_nosoc
    drho = -(tr - np.conjugate(tr[:, neg][:, neg])) / (2j*np.pi)
    drho_r = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(drho, axes=(1, 2))), axes=(1, 2))
    zero_tol = 1e-10 * v_mag*np.amax(np.sum(kws**2, axis=-1)) * np.sum(np.abs(g0)**2)
    np.savez(path, drho=drho, drho_r=drho_r, zero_tol=zero_tol, omega=OMEGA_NODAL)
    return path


def files(imp, folder):
    """[omega = 0 file, omega_2 file] for one impurity and pairing."""
    sub = 'q0_1.00e+00' if imp == 'A1' else 'q0_2.00e+01'
    fs = sorted(glob.glob(os.path.join(NOSOC, imp, sub, folder, 'drho_*.npz')))
    if folder != 'dx2y2':
        return fs
    nodal = (glob.glob(os.path.join(FULLT, f'drho_om{OMEGA_NODAL:.4f}_*.npz'))[0] if imp == 'A1'
             else a2_dx2y2_nodal())
    return [fs[0], nodal]


def panel(a, kind, img, geom, zero_tol, r_window):
    """Fig. 4 style: normalised map, max below, no ticks or box; '~0' if below zero_tol."""
    vmax = np.amax(np.abs(img))
    zero = vmax < zero_tol
    draw_panel(a, kind, None if zero else img/vmax, *geom[kind], r_window)
    if not zero:
        a.set_xlabel(f'{vmax:.2g}', labelpad=1, fontsize=8)
    a.set_xticks([])
    a.set_yticks([])
    a.spines[:].set_visible(False)


def col_titles(ax):
    ax[0, 0].set_title(r'$\delta\rho(\bvec{q})$', fontsize=12)
    ax[0, 1].set_title(r'$\delta\rho(\bvec{r})$', fontsize=12)


def row_label(a, i):
    """(a), (b), ... at the top left of row i, quoted in the case table."""
    a.text(-0.42, 1.0, f'({"abcdefgh"[i]})', transform=a.transAxes, ha='right', va='top', fontsize=12)


def bottom_colorbar(sf):
    sf.colorbar(ScalarMappable(NORM, CMAP), cax=sf.add_axes([0.35, 0.7, 0.3, 0.12]),
                orientation='horizontal', ticks=[-1, 0, 1], label=r'value / $\max|\cdot|$')


def combined(imp, path, Nq=120, r_window=10):
    """Column pairs omega = 0 and omega_2; rows = pairings, separated by horizontal lines."""
    geom = dict(q=(setup_mesh(Nq, rlattice), rlattice), r=(setup_mesh(Nq, lattice) * Nq, lattice))
    fig = plt.figure(figsize=(5.6, 6.4), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.09])
    subs = top.subfigures(1, 2, wspace=0.03, width_ratios=[1.2, 1])
    axs = []
    for j, (sf, title) in enumerate(zip(subs, [r'$\omega = 0$', r'$\omega = \omega_2$'])):
        sf.suptitle(title, fontsize=12)
        sf.get_layout_engine().set(h_pad=0.01, hspace=0)
        axs.append(sf.subplots(len(PAIRINGS), 2))
        col_titles(axs[j])
    for i, (folder, _, tex) in enumerate(PAIRINGS):
        for j, f in enumerate(files(imp, folder)):
            d = np.load(f)
            # only Re drho_0 is finite without SOC; Im drho_0(q) and spin channels are round-off
            for c, (kind, img) in enumerate([('q', np.real(d['drho'][0])), ('r', np.real(d['drho_r'][0]))]):
                panel(axs[j][i, c], kind, img, geom, d['zero_tol'], r_window)
        axs[0][i, 0].set_ylabel(tex + '\n' + rf'$\omega_2 = {float(d["omega"]):.4f}$', fontsize=11)
        row_label(axs[0][i, 0], i)
    bottom_colorbar(bottom)
    fig.canvas.draw()  # fix the layout, then place the dividers between rows
    inv = top.transSubfigure.inverted()
    for i in range(len(PAIRINGS) - 1):
        lo = inv.transform(axs[0][i, 0].get_tightbbox().p0)[1]
        hi = inv.transform(axs[0][i + 1, 0].get_tightbbox().p1)[1]
        y = (lo + hi) / 2
        top.add_artist(Line2D([0.01, 0.99], [y, y], transform=top.transSubfigure, color='0.5', lw=0.8))
    fig.set_layout_engine('none')  # keep the measured layout on save
    fig.savefig(path, dpi=300)
    plt.close(fig)


def nonzero_only(imp, path, Nq=120, r_window=10):  # column width (3.4 in)
    """One row per (pairing, omega) whose Re drho_0 is finite: (Re drho(q), drho(r))."""
    geom = dict(q=(setup_mesh(Nq, rlattice), rlattice), r=(setup_mesh(Nq, lattice) * Nq, lattice))
    rows = []
    for folder, _, tex in PAIRINGS:
        for f in files(imp, folder):
            d = np.load(f)
            if np.amax(np.abs(np.real(d['drho'][0]))) >= d['zero_tol']:
                rows.append((tex, d))
    fig = plt.figure(figsize=(3.4, 1.55*len(rows) + 0.9), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.55*len(rows) + 0.35, 0.55])
    top.get_layout_engine().set(h_pad=0.01, hspace=0)
    ax = top.subplots(len(rows), 2, squeeze=False)
    for i, (tex, d) in enumerate(rows):
        for c, (kind, img) in enumerate([('q', np.real(d['drho'][0])), ('r', np.real(d['drho_r'][0]))]):
            panel(ax[i, c], kind, img, geom, d['zero_tol'], r_window)
        ax[i, 0].set_ylabel(tex + '\n' + rf'$\omega = {float(d["omega"]):.4f}$', fontsize=11)
    col_titles(ax)
    bottom_colorbar(bottom)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    combined('A1', os.path.join(out, 'nosoc_A1imp_all.pdf'))
    combined('A2', os.path.join(out, 'nosoc_A2imp_all.pdf'))
    nonzero_only('A2', os.path.join(out, 'nosoc_A2imp_nonzero.pdf'))
