# Combined no-SOC figures, one per impurity: rows = pairings, columns = (Re drho(q), drho(r)) at omega = 0 and omega_2.
# omega_2 = 0.98 gap for gapped states, 0.15 for nodal dx2y2. Only Re drho_0 is non-zero without SOC.
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2026-11-05_nosoc'))
from qpi_nosoc import a2_S  # noqa: E402  (also puts 2_new on sys.path)
from qpi_simulation import (PAULI, setup_mesh, neg_index, fold_to_ws, build_c6v_perms, calculate_bloch,  # noqa: E402
                            calculate_sc_order_params, calculate_bdg, calculate_greens_function,
                            irrep_decomposition, draw_panel, corner_note, add_colorbar, IRREP_NAMES)

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


def weights_label(img, mesh, U, min_weight=1e-3):
    _, w, _ = irrep_decomposition(img, mesh, U)
    return ', '.join(rf'{IRREP_NAMES[n][0]}$_{IRREP_NAMES[n][1]}$ {100*w[n]:.0f}\%' for n in np.argsort(-w) if w[n] > min_weight)


def combined(imp, path, Nq=120, r_window=10):
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq = build_c6v_perms(qmesh)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))
    fig, ax = plt.subplots(len(PAIRINGS), 4, figsize=(6.4, 6.0), layout='constrained')
    for i, (folder, _, tex) in enumerate(PAIRINGS):
        for j, f in enumerate(files(imp, folder)):
            d = np.load(f)
            # only Re drho_0 is finite without SOC; Im drho_0(q) and spin channels are round-off
            for c, (kind, img) in enumerate([('q', np.real(d['drho'][0])), ('r', np.real(d['drho_r'][0]))]):
                a = ax[i, 2*j + c]
                vmax = np.amax(np.abs(img))
                zero = vmax < d['zero_tol']
                draw_panel(a, kind, None if zero else img/vmax, *geom[kind], r_window)
                if not zero:
                    corner_note(a, f'{vmax:.2g}')
                    if kind == 'q':
                        a.set_xlabel(weights_label(img, qmesh, Uq), labelpad=1)
                a.set_xticks([])
                a.set_yticks([])
                if i == 0:
                    a.set_title((r'$\mathrm{Re}\,\delta\rho(\bvec{q})$' if kind == 'q' else r'$\delta\rho(\bvec{r})$')
                                + (r', $\omega = 0$' if j == 0 else r', $\omega_2$'))
            if j == 1:
                om2 = float(d['omega'])
        ax[i, 0].set_ylabel(tex + '\n' + rf'$\omega_2 = {om2:.4f}$')
    add_colorbar(fig, ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def nonzero_only(imp, path, Nq=120, r_window=10):  # column width (3.4 in)
    """One row per (pairing, omega) whose Re drho_0 is finite: (Re drho(q), drho(r))."""
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq = build_c6v_perms(qmesh)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))
    rows = []
    for folder, _, tex in PAIRINGS:
        for f in files(imp, folder):
            d = np.load(f)
            if np.amax(np.abs(np.real(d['drho'][0]))) >= d['zero_tol']:
                rows.append((tex, d))
    fig, ax = plt.subplots(len(rows), 2, figsize=(3.4, 1.75*len(rows)), layout='constrained', squeeze=False)
    for i, (tex, d) in enumerate(rows):
        for c, (kind, img) in enumerate([('q', np.real(d['drho'][0])), ('r', np.real(d['drho_r'][0]))]):
            a = ax[i, c]
            vmax = np.amax(np.abs(img))
            draw_panel(a, kind, img/vmax, *geom[kind], r_window)
            corner_note(a, f'{vmax:.2g}')
            if kind == 'q':
                a.set_xlabel(weights_label(img, qmesh, Uq), labelpad=1)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title(r'$\mathrm{Re}\,\delta\rho(\bvec{q})$' if kind == 'q' else r'$\delta\rho(\bvec{r})$')
        ax[i, 0].set_ylabel(tex + '\n' + rf'$\omega = {float(d["omega"]):.4f}$')
    add_colorbar(fig, ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figs')
    os.makedirs(out, exist_ok=True)
    combined('A1', os.path.join(out, 'nosoc_A1imp_all.pdf'))
    combined('A2', os.path.join(out, 'nosoc_A2imp_all.pdf'))
    nonzero_only('A2', os.path.join(out, 'nosoc_A2imp_nonzero.pdf'))
