# Normal, BdG and zoomed BdG bands on Gamma-M-K-Gamma: Rashba d+id (ansatz) and the chiral FRG gap (U = 8), one column each.
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import diverge.output as do  # type:ignore

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
import qpi_simulation as rs  # noqa: E402  (sets the mpl style)
import qpi_simulation_frg as frg  # noqa: E402

DELTA = 0.1     # |Delta| (Rashba) and Delta_rms (FRG), as the QPI runs
ZOOM = 4        # zoom row shows |E| < ZOOM x gap
lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


def rashba(k, mu=-0.1, alpha=0.1):
    h = rs.calculate_bloch(mu, alpha, k, lattice)
    return h, rs.calculate_bdg(h, rs.calculate_bloch(mu, alpha, -k, lattice),
                               DELTA*rs.calculate_sc_order_params(k, lattice, 'd+id'))


def frg_model(run_dir=frg.RUN_DIR, chirality=1):
    """BdG builder as qpi_simulation_frg.main."""
    name = os.path.basename(os.path.normpath(run_dir))
    O = do.read(os.path.join(run_dir, name + '.dvg'))
    M = do.read(os.path.join(run_dir, name + '.mod'))
    latt = np.array(M.lattice)[:2, :2]
    bond = frg.get_bond_vectors(O, latt)[:, 0, 0, :]
    inp = os.path.join(run_dir, name + '.input')
    ff, t, _ = frg.bloch_ff_from_w90(os.path.join(run_dir, frg.read_input_key(inp, 'tbfile')))
    mu = float(frg.read_input_key(inp, 'chemical_potential'))

    def bdg(k):
        h = frg.calculate_bloch(ff, t, mu, k, latt)
        ev = frg.calculate_sc_order_params(O, k, bond)
        delt = ev[0] if len(ev) == 1 else (ev[0] + 1j*chirality*ev[1])/np.sqrt(2)
        return h, frg.calculate_bdg(h, frg.calculate_bloch(ff, t, mu, -k, latt), DELTA*delt)
    return bdg


def bands(bdg, kpath):
    """Normal and BdG bands on the path, and the BZ gap on a 240^2 mesh."""
    h, H = bdg(kpath[None])
    gap = np.amin(np.abs(np.linalg.eigvalsh(bdg(rs.setup_mesh(240, rlattice))[1])))
    return np.linalg.eigvalsh(h)[0], np.linalg.eigvalsh(H)[0], gap


def zoom_window(dist, normal, E, gap):
    """k-range around the normal-state crossings nearest the smallest path gap, padded so |E| < ZOOM gap is resolved."""
    i0 = np.argmin(np.amin(np.abs(E), axis=1))
    cross = np.where(np.diff(np.sign(normal), axis=0))[0]
    cross = cross[np.abs(dist[cross] - dist[i0]) < 0.5]
    slope = np.abs(np.gradient(normal, dist, axis=0))[i0].min()
    pad = 1.5*ZOOM*gap/slope
    return dist[cross].min() - pad, dist[cross].max() + pad


def main():
    kpath, dist, ticks, labels = rs.high_symmetry_path(rlattice, n=600)
    cases = [(r'Rashba, $d+id$', rashba), (r'FRG, $\chi = +1$', frg_model())]
    fig, ax = plt.subplots(3, 2, figsize=(3.4, 3.4), layout='constrained')
    for j, (title, bdg) in enumerate(cases):
        normal, E, gap = bands(bdg, kpath)
        print(f'{title}: gap {gap:.4f}')
        ax[0, j].plot(dist, normal, color=rs.INK, lw=0.8)
        ax[0, j].set_title(title)
        for a in ax[1:, j]:
            a.plot(dist, normal, color=rs.INK_MUTED, lw=0.6, ls='--')
            a.plot(dist, -normal, color=rs.INK_MUTED, lw=0.6, ls='--')
            a.plot(dist, E, color=rs.ACCENT, lw=0.9)
        x0, x1 = zoom_window(dist, normal, E, gap)
        ax[2, j].set_xlim(x0, x1)
        ax[2, j].set_ylim(-ZOOM*gap, ZOOM*gap)
        ax[1, j].add_patch(Rectangle((x0, -ZOOM*gap), x1 - x0, 2*ZOOM*gap, fill=False, ec=rs.INK, lw=0.6))
        ax[2, j].axhspan(-gap, gap, color=rs.ACCENT, alpha=0.12, lw=0)
        ax[2, j].text(0.97, 0.95, rf'$E_g = {gap:.3f}$', transform=ax[2, j].transAxes,
                      ha='right', va='top', fontsize=7, bbox=dict(fc='white', ec='none', pad=1))
    for a in ax.flat:
        a.axhline(0, color=rs.INK_MUTED, lw=0.4)
    for a in ax[:2].flat:
        for t in ticks[1:-1]:
            a.axvline(t, color=rs.INK_MUTED, lw=0.4)
        a.set_xticks(ticks, labels if a in ax[1] else [])
        a.set_xlim(dist[0], dist[-1])
    for j in range(2):
        seg = np.searchsorted(ticks, np.mean(ax[2, j].get_xlim())) - 1
        ax[2, j].set_xticks([])
        ax[2, j].set_xlabel(f'{labels[seg]}--{labels[seg + 1]}')
    for i, lab in enumerate(['Normal', 'BdG', 'BdG (zoom)']):
        ax[i, 0].set_ylabel(rf'{lab}: $E$')
    os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'figs', 'bands_combined.pdf'))
    fig.savefig(os.path.join(HERE, 'figs', 'bands_combined.png'), dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
