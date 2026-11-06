# Is the dx2y2 (and ext s) ansatz state gapped? Writes nodes.txt.
# Singlet pairing + Rashba decouples into helical bands, E_pm = sqrt(xi_pm^2 + |Psi|^2), xi_pm = eps +- alpha |grad eps|,
# so the state is gapless iff a zero line of Psi crosses a helical Fermi surface xi_pm = 0.
#  1. helical formula vs full BdG;  2. min|E| on the QPI k-mesh vs Nk;  3. continuum min|E| by grid zooming;
#  4. Fermi surfaces by radial bisection, nodes = sign changes of Psi along them, and the max |Delta| on them.
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
from qpi_simulation import setup_mesh, calculate_bloch, calculate_sc_order_params, calculate_bdg  # noqa: E402

MU, DMAG = -0.1, 0.1
LATT = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
B = 2*np.pi*np.linalg.inv(LATT).T
NN = np.array([LATT[0], LATT[1], LATT[0] + LATT[1]])


def xi(k, s, alpha):
    ph = k @ NN.T
    return 2*np.cos(ph).sum(-1) + MU + s*alpha*np.linalg.norm(-2*np.sin(ph) @ NN, axis=-1)


def psi(k, pairing):
    return DMAG*calculate_sc_order_params(k.reshape(1, -1, 2), LATT, pairing).reshape(k.shape[:-1])


def bands(k, pairing, alpha):
    return [np.sqrt(xi(k, s, alpha)**2 + np.abs(psi(k, pairing))**2) for s in (1, -1)]


def continuum_min(pairing, alpha, N=1920, n_start=20, n_zoom=60):
    """Zoom a 41x41 grid (factor 4 per step) around the n_start lowest mesh points of each band."""
    km = setup_mesh(N, B)
    best = (np.inf, None, None)
    for s in range(2):
        e = bands(km, pairing, alpha)[s]
        for i in np.argsort(e.ravel())[:n_start]:
            x, h = km.reshape(-1, 2)[i], np.linalg.norm(B[0])/N
            for _ in range(n_zoom):
                g = x + h*np.stack(np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41)), -1).reshape(-1, 2)
                v = bands(g, pairing, alpha)[s]
                x, h = g[np.argmin(v)], h/4
            if v.min() < best[0]:
                best = (v.min(), '+-'[s], x)
    return best


def fermi_nodes(pairing, alpha, s, n_th=72000):
    """Radial bisection for xi_s = 0 at n_th angles; returns node angles (deg) and max |Delta| on the FS."""
    th = np.linspace(0, 2*np.pi, n_th, endpoint=False)
    d = np.stack([np.cos(th), np.sin(th)], -1)
    r = np.linspace(0, 3.5, 2000)
    vals = np.array([xi(d*ri, s, alpha) for ri in r])
    idx = np.argmax(np.sign(vals) != np.sign(vals[0]), axis=0)
    lo, hi = r[idx-1], r[idx]
    for _ in range(60):
        m = (lo + hi)/2
        same = np.sign(xi(d*m[:, None], s, alpha)) == np.sign(vals[0])
        lo, hi = np.where(same, m, lo), np.where(same, hi, m)
    p = psi(d*lo[:, None], pairing).real
    nodes = np.where(np.sign(p) != np.sign(np.roll(p, -1)))[0]
    return np.degrees(th[nodes]), np.abs(p).max()


if __name__ == '__main__':
    with open(os.path.join(HERE, 'nodes.txt'), 'w') as out:
        for pairing in ['dx2y2', 'ext s']:
            out.write(f'== {pairing}, mu={MU}, |Delta|={DMAG}\n')
            k = setup_mesh(60, B)
            en = np.linalg.eigvalsh(calculate_bdg(calculate_bloch(MU, 0.1, k, LATT), calculate_bloch(MU, 0.1, -k, LATT),
                                                  DMAG*calculate_sc_order_params(k, LATT, pairing)))
            e = np.stack(bands(k, pairing, 0.1), -1)
            out.write(f'  helical formula vs BdG (alpha=0.1, Nk=60): max diff {np.abs(en - np.sort(np.concatenate([e, -e], -1), -1)).max():.1e}\n')
            for alpha in [0.1, 0.0]:
                for N in [60, 120, 240, 480, 960, 1920]:
                    out.write(f'  alpha={alpha}, Nk={N:5d}: min|E| on mesh = {min(x.min() for x in bands(setup_mesh(N, B), pairing, alpha)):.3e}\n')
                f, band, x = continuum_min(pairing, alpha)
                out.write(f'  alpha={alpha}: continuum min|E| = {f:.2e} on band {band} at k = {np.round(x, 5)}\n')
                for s in ([1, -1] if alpha else [1]):
                    ang, dmax = fermi_nodes(pairing, alpha, s)
                    out.write(f'  alpha={alpha}, FS xi_{"+-"[s < 0]}: {len(ang)} nodes at {np.round(ang, 1)} deg, max|Delta| on FS = {dmax:.4f}\n')
    print(open(os.path.join(HERE, 'nodes.txt')).read())
