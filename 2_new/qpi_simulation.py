# QPI from an A1 impurity in the Born approximation, with nearest-neighbour Rashba SOC
import os
import numpy as np
from tqdm import tqdm #type:ignore
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.path import Path

mpl.rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern Roman'], 'size': 8})
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}'
       + r'\usepackage{braket}'
       + r'\usepackage{amssymb}'
       + r'\newcommand{\bvec}[1]{\boldsymbol{#1}}')

HERE = os.path.dirname(os.path.abspath(__file__))

CMAP, NORM = 'RdBu_r', Normalize(-1, 1)
INK, INK_MUTED, ACCENT = '#0b0b0b', '#52514e', '#2a78d6'

PAULI = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
CHANNELS = ['0', 'x', 'y', 'z']


##############
# Meshes and C6v symmetry
##############

def setup_mesh(N, b):
    """Centred N x N mesh of (n1 b[0] + n2 b[1])/N; axis 0 steps along b[1], axis 1 along b[0]."""
    n = np.arange(N) - N//2
    n1, n2 = np.meshgrid(n, n)
    return (n1[..., None]*b[0] + n2[..., None]*b[1]) / N

def neg_index(N):
    """Index of -k on the centred mesh, per axis."""
    return (N - np.arange(N)) % N

def fold_to_ws(mesh, b, tol=1e-9):
    """Fold mesh into the Wigner-Seitz cell, averaging boundary points over their equivalent images."""
    shifts = np.array([i*b[0] + j*b[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    cand = mesh[..., None, :] + shifts
    d = np.sum(cand**2, axis=-1)
    w = d <= np.amin(d, axis=-1, keepdims=True) + tol
    return np.sum(cand * w[..., None], axis=-2) / np.sum(w, axis=-1)[..., None]

def build_c6v_perms(kmesh, tol=1e-6):
    """
    U[g][i] is the flat index of g(k[i]), so H[U[g]] is the pullback H(g k).
    Order: [E, C6, ..., C6^5, s, C6 s, ..., C6^5 s], s: ky -> -ky.
    """
    N = kmesh.shape[0]
    k = kmesh.reshape(-1, 2)
    steps = np.array([kmesh[0,1] - kmesh[0,0], kmesh[1,0] - kmesh[0,0]])

    def perm_from_matrix(R):
        n = np.linalg.solve(steps.T, (k @ R.T - kmesh[0,0]).T).T
        n_int = np.rint(n)
        if np.amax(np.abs(n - n_int)) > tol:
            raise ValueError('Symmetry operation does not map the k-mesh onto itself')
        n_int = n_int.astype(int) % N
        return n_int[:,1]*N + n_int[:,0]

    th = np.pi/3
    c6 = perm_from_matrix(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    s = perm_from_matrix(np.diag([1.0, -1.0]))

    U = [np.arange(N*N)]
    for _ in range(5):
        U.append(c6[U[-1]])
    U += [U[n][s] for n in range(6)]
    return np.array(U)

IRREP_NAMES = ['A1', 'A2', 'B1', 'B2', 'E1', 'E2']

def c6v_character_table():
    """Rows as IRREP_NAMES, columns in the element order of build_c6v_perms."""
    return np.array([
        [1,  1,  1,  1,  1,  1,    1,  1,  1,  1,  1,  1],
        [1,  1,  1,  1,  1,  1,   -1, -1, -1, -1, -1, -1],
        [1, -1,  1, -1,  1, -1,    1, -1,  1, -1,  1, -1],
        [1, -1,  1, -1,  1, -1,   -1,  1, -1,  1, -1,  1],
        [2,  1, -1, -2, -1,  1,    0,  0,  0,  0,  0,  0],
        [2, -1, -1,  2, -1, -1,    0,  0,  0,  0,  0,  0],
    ])

def project_to_irreps(H, kmesh, U=None):
    """H_nu = (d_nu/|G|) sum_g chi_nu(g) H[U[g]]; H is (N^2, ...) or (N, N, ...), returns (6, *H.shape)."""
    if U is None:
        U = build_c6v_perms(kmesh)
    N = kmesh.shape[0]
    if H.shape[0] == N*N:
        trailing = H.shape[1:]
    elif H.shape[:2] == (N, N):
        trailing = H.shape[2:]
    else:
        raise ValueError(f'H has shape {H.shape}; expected k-axes first, as ({N*N}, ...) or ({N}, {N}, ...)')
    Hf = H.reshape(N*N, *trailing)

    chi = c6v_character_table()
    coef = chi * chi[:, :1] / len(U)

    Hnu = 0
    for g in range(len(U)):
        Hnu = Hnu + np.multiply.outer(coef[:, g], Hf[U[g]])
    return Hnu.reshape(len(chi), *H.shape)


##############
# Model
##############

def calculate_bloch(mu, alpha, bz, latt):
    """Spin Bloch matrix h(k) = eps(k) + alpha (d_ky eps sigma_x - d_kx eps sigma_y), shape (N, N, 2, 2)."""
    nn = np.array([latt[0], latt[1], latt[0] + latt[1]])
    phase = bz @ nn.T
    eps = 2*np.sum(np.cos(phase), axis=-1) + mu
    grad = -2*np.sin(phase) @ nn
    h = np.zeros(bz.shape[:2] + (2, 2), dtype=complex)
    h[..., 0, 0] = eps
    h[..., 1, 1] = eps
    h[..., 0, 1] = alpha*(grad[..., 1] + 1j*grad[..., 0])
    h[..., 1, 0] = np.conjugate(h[..., 0, 1])
    return h

def calculate_sc_order_params(bz, latt, pairing):
    """Singlet form factor Psi(k), unnormalised."""
    if pairing == 'd+id':
        d = np.array([np.exp(1j*0), np.exp(1j*2*np.pi/3), np.exp(1j*4*np.pi/3)])
    elif pairing == 'dx2y2':
        d = np.array([1, -0.5, -0.5])
    elif pairing == 'ext s':
        d = np.array([1, 1, 1])
    elif pairing == 's':
        return np.ones_like(bz[:,:,0])
    else:
        raise ValueError('Pairing not recognized')

    return 2*(
        d[0]*np.cos(np.einsum('i,jki',latt[0,:],bz)) +
        d[1]*np.cos(np.einsum('i,jki',latt[1,:],bz)) +
        d[2]*np.cos(np.einsum('i,jki',latt[0,:]+latt[1,:],bz))
    )

def calculate_bdg(h, h_neg, op):
    """Basis (c_k up, c_k dn, c+_-k up, c+_-k dn): [[h(k), Delta(k)], [Delta(k)^+, -h(-k)^T]], Delta = Psi i sigma_y."""
    delta = op[..., None, None] * (1j*PAULI[2])
    bdg_ham = np.zeros(h.shape[:-2] + (4, 4), dtype=complex)
    bdg_ham[..., :2, :2] = h
    bdg_ham[..., 2:, 2:] = -np.swapaxes(h_neg, -1, -2)
    bdg_ham[..., :2, 2:] = delta
    bdg_ham[..., 2:, :2] = np.conjugate(np.swapaxes(delta, -1, -2))
    return bdg_ham

def calculate_greens_function(omega, gamma, en, u):
    """Retarded g0(k, omega), shape (len(omega), N, N, 4, 4)."""
    return np.einsum('jkab,ojkb,jkbc->ojkac',
        u,
        1/(np.reshape(omega,(len(omega),1,1,1)) + 1j*gamma - en),
        np.conjugate(np.swapaxes(u,2,3))
    )

def born_S(g0, T, Nq, qstep):
    """S(q) = sum_k g0(k) T g0(k-q) on the centred q-mesh, for a q-independent T."""
    gT = g0 @ T
    nb = g0.shape[-1]
    S = np.zeros((Nq,Nq,nb,nb), dtype=complex)
    for i, j in tqdm(np.ndindex(Nq, Nq), total=Nq*Nq, desc='QPI loop'):
        shift = ((i - Nq//2)*qstep, (j - Nq//2)*qstep)
        S[i,j] = np.tensordot(gT, np.roll(g0, shift, axis=(0,1)), axes=([0,1,3], [0,1,2]))
    return S


##############
# Analysis and plotting
##############

def irrep_decomposition(f, mesh, U):
    """Returns components fnu, weight fractions sum|f_nu|^2 / sum|f|^2, and max |f_nu|."""
    fnu = project_to_irreps(f, mesh, U)
    total = np.sum(np.abs(f)**2)
    weights = np.sum(np.abs(fnu)**2, axis=(1,2)) / total if total > 0 else np.zeros(len(fnu))
    return fnu, weights, np.amax(np.abs(fnu), axis=(1,2))

def write_irrep_analysis(f, label, weights, maxvals, negligible=False):
    if negligible:
        f.write(f'{label}: numerically zero (max |component| = {np.amax(maxvals):.4e}), weights not meaningful\n')
        return
    dom = np.argmax(weights)
    f.write(f'{label}: dominant irrep {IRREP_NAMES[dom]}, max |component| = {maxvals[dom]:.4e}\n')
    for name, w, m in zip(IRREP_NAMES, weights, maxvals):
        f.write(f'    {name}: weight = {w:.6f}, max |component| = {m:.4e}\n')

def ws_hexagon(c1, c2):
    """Vertices of the Wigner-Seitz hexagon of the triangular lattice spanned by c1, c2."""
    nn = np.array([c1, c2, c1 + c2, c1 - c2])
    nn = np.concatenate([nn, -nn])
    nn = nn[np.argsort(np.linalg.norm(nn, axis=1))[:6]]
    nn = nn[np.argsort(np.arctan2(nn[:,1], nn[:,0]))]
    return (nn + np.roll(nn, -1, axis=0)) / 3

def draw_panel(ax, kind, z, mesh, cell, r_window):
    """
    z normalised to [-1, 1], or None for a blank panel. kind 'q': periodic data tiled into the
    first BZ of cell; kind 'r': lattice sites as Wigner-Seitz cells within a hexagon of radius r_window.
    """
    if kind == 'q':
        outline = ws_hexagon(*cell)
    else:
        th = np.pi/3 * np.arange(6)
        outline = r_window * np.stack([np.cos(th), np.sin(th)], axis=1)
    border = ax.add_patch(Polygon(outline, fill=False, edgecolor=INK_MUTED, lw=0.5, zorder=3))
    if z is None:
        ax.text(0.5, 0.5, r'$\approx 0$', transform=ax.transAxes, ha='center', va='center', color=INK_MUTED)
    elif kind == 'q':
        N = z.shape[0]
        idx = (np.arange(2*N) - N//2) % N
        ext = setup_mesh(2*N, 2*cell)
        im = ax.pcolormesh(ext[...,0], ext[...,1], z[np.ix_(idx, idx)], cmap=CMAP, norm=NORM,
                           shading='nearest', rasterized=True)
        im.set_clip_path(border)
    else:
        sites = mesh.reshape(-1, 2)
        inside = Path(outline*1.01).contains_points(sites)
        im = PolyCollection(sites[inside, None, :] + ws_hexagon(*cell), array=z.reshape(-1)[inside],
                            cmap=CMAP, norm=NORM, edgecolors='face', linewidths=0.2, rasterized=True)
        ax.add_collection(im)
        im.set_clip_path(border)
    lim = 1.03*np.amax(np.abs(outline), axis=0)
    ax.set_xlim(-lim[0], lim[0])
    ax.set_ylim(-lim[1], lim[1])
    ax.set_aspect('equal')

def corner_note(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha='left', va='top', color=INK_MUTED)

def add_colorbar(fig, ax):
    fig.colorbar(ScalarMappable(NORM, CMAP), ax=ax, shrink=0.4, ticks=[-1, 0, 1],
                 label=r'value / $\max|\cdot|$')

def plot_drho(datasets, geom, r_window, path, title):
    """Rows: spin channels; columns: Re drho(q), Im drho(q), drho(r). Each panel scaled to its own max."""
    fig, ax = plt.subplots(len(CHANNELS), 3, figsize=(5.5, 7), layout='constrained')
    for i, d in enumerate(datasets):
        a = ax[i // 3, i % 3]
        vmax = np.amax(np.abs(d['img']))
        draw_panel(a, d['kind'], None if d['negligible'] else d['img']/vmax, *geom[d['kind']], r_window)
        a.set_title(d['label'])
        if not d['negligible']:
            corner_note(a, f'{vmax:.2g}')
        a.tick_params(labelbottom=(i // 3 == len(CHANNELS) - 1), labelleft=(i % 3 != 1))
    for col, (xl, yl) in enumerate([(r'$q_x$', r'$q_y$'), (r'$q_x$', ''), (r'$x$', r'$y$')]):
        ax[-1, col].set_xlabel(xl)
        for a in ax[:, col]:
            a.set_ylabel(yl)
    add_colorbar(fig, ax)
    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_irreps(datasets, geom, r_window, path, title, min_weight=1e-4):
    """
    One row per non-zero dataset: total, then each irrep with weight in some row, all scaled to
    the total's max. Irreps without weight in a row are left blank.
    """
    rows = [d for d in datasets if not d['negligible']]
    if not rows:
        return
    cols = [nu for nu in range(len(IRREP_NAMES)) if any(d['weights'][nu] > min_weight for d in rows)]
    fig, ax = plt.subplots(len(rows), 1 + len(cols), figsize=(1.05*(1 + len(cols)) + 1, 1.05*len(rows) + 0.7),
                           layout='constrained', squeeze=False)
    for i, d in enumerate(rows):
        vmax = np.amax(np.abs(d['img']))
        for j, nu in enumerate([None] + cols):
            a = ax[i, j]
            if nu is None:
                z, note = d['img']/vmax, f'{vmax:.2g}'
            elif d['weights'][nu] > min_weight:
                z, note = d['fnu'][nu]/vmax, rf'{100*d["weights"][nu]:.1f}\%'
            else:
                a.set_axis_off()
                continue
            draw_panel(a, d['kind'], z, *geom[d['kind']], r_window)
            a.set_xlabel(note, labelpad=1)
            a.set_xticks([])
            a.set_yticks([])
            a.set_frame_on(False)
        ax[i, 0].set_ylabel(d['label'])
    for j, name in enumerate(['total'] + [IRREP_NAMES[nu] for nu in cols]):
        ax[0, j].set_title(name)
    add_colorbar(fig, ax)
    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)

def high_symmetry_path(b, n=150):
    """Gamma-M-K-Gamma: k points, cumulative distance, node positions and labels."""
    nodes = np.array([np.zeros(2), b[1]/2, (b[0] + b[1])/3, np.zeros(2)])
    k = np.concatenate([np.linspace(p, q, n, endpoint=False) for p, q in zip(nodes[:-1], nodes[1:])] + [nodes[-1:]])
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(k, axis=0), axis=1))])
    return k, dist, dist[::n], [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$']

def plot_bands(dist, ticks, labels, normal, bdg, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True, layout='constrained')
    ax[0].plot(dist, normal, color=INK, lw=1)
    ax[0].set_title('Normal state')
    ax[1].plot(dist, normal, color=INK_MUTED, lw=0.8, ls='--')
    ax[1].plot(dist, -normal, color=INK_MUTED, lw=0.8, ls='--')
    ax[1].plot(dist, bdg, color=ACCENT, lw=1.2)
    ax[1].set_title(r'BdG (dashed: $\pm\varepsilon(\bvec{k})$)')
    for a in ax:
        a.axhline(0, color=INK_MUTED, lw=0.5)
        for t in ticks[1:-1]:
            a.axvline(t, color=INK_MUTED, lw=0.5)
        a.set_xticks(ticks, labels)
        a.set_xlim(dist[0], dist[-1])
    ax[0].set_ylabel(r'$E$')
    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


##############
# Simulation
##############

def main(pairing, mu=-0.1, alpha=0.1, delta_mag=1.0, gamma=1e-2, v_mag=1e-1, Nk=120, Nq=120,
         q0_factors=10.0**np.linspace(-2, 2, 9), r_window=10):
    """q0 = q0_factor * |b1|; r_window is the plotted real-space radius in lattice constants."""

    if Nk % Nq != 0:
        raise ValueError('Nk must be a multiple of Nq so k-q lies on the k-mesh')
    qstep = Nk // Nq

    sav_dir = os.path.join(HERE, 'data', f'delta{delta_mag:g}_gamma{gamma:g}_alpha{alpha:g}')
    pairing_name = pairing.replace('+', 'p').replace(' ', '_')
    os.makedirs(sav_dir, exist_ok=True)

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(Nk, rlattice)
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq = build_c6v_perms(qmesh)
    Ur = build_c6v_perms(rmesh)
    neg = neg_index(Nq)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))

    # V depends on |q|, which is not periodic, so evaluate it on the Wigner-Seitz cell
    q_sq = np.sum(fold_to_ws(qmesh, rlattice)**2, axis=-1)

    h = calculate_bloch(mu, alpha, kmesh, lattice)
    op = delta_mag * calculate_sc_order_params(kmesh, lattice, pairing)
    en, u = np.linalg.eigh(calculate_bdg(h, calculate_bloch(mu, alpha, -kmesh, lattice), op))
    gap_size = np.amin(np.abs(en))
    print(f'{pairing}: gap size {gap_size}')

    kpath, dist, ticks, labels = high_symmetry_path(rlattice)
    h_path = calculate_bloch(mu, alpha, kpath[None], lattice)
    en_path = np.linalg.eigvalsh(calculate_bdg(
        h_path, calculate_bloch(mu, alpha, -kpath[None], lattice),
        delta_mag * calculate_sc_order_params(kpath[None], lattice, pairing)))
    plot_bands(dist, ticks, labels, np.linalg.eigvalsh(h_path)[0], en_path[0],
               os.path.join(sav_dir, f'bands_{pairing_name}.png'),
               rf'{pairing} pairing, $\alpha$ = {alpha:g}, $\mu$ = {mu:g}, $|\Delta|$ = {delta_mag:g}, gap = {gap_size:.4f}')
    if gap_size < 1e-6:
        with open(os.path.join(sav_dir, f'no_gap_{pairing_name}.txt'), 'w') as f:
            f.write(f'No gap found for mu {mu}, alpha {alpha}, gamma {gamma}, delta_mag {delta_mag}\n')
        return

    omega_vals = np.array([0, 0.98*gap_size])
    g0k = calculate_greens_function(omega_vals, gamma, en, u)
    T = v_mag * np.diag([1, 1, -1, -1])

    for ii, om in enumerate(omega_vals):

        S_inf = born_S(g0k[ii], T, Nq, qstep)
        zero_tol = 1e-10 * v_mag * np.sum(np.abs(g0k[ii])**2)

        for q0_factor in q0_factors:

            q0 = np.linalg.norm(rlattice[0])*q0_factor
            out_dir = os.path.join(sav_dir, f'q0_{q0_factor:.2e}', pairing_name)
            os.makedirs(out_dir, exist_ok=True)
            tag = f'om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}'
            title = (
                rf'{pairing} pairing, $\alpha$ = {alpha:g}, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}'
                '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b_1|$'
            )

            S = S_inf * np.exp(-q_sq / q0**2)[:, :, None, None]

            # drho_mu(q) = -[Tr sigma_mu S(q) - Tr sigma_mu S(-q)^+]/(2 pi i), particle block only
            tr = np.einsum('mab,ijba->mij', PAULI, S[..., :2, :2])
            drho = -(tr - np.conjugate(tr[:, neg][:, :, neg])) / (2j*np.pi)
            drho_r = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(drho, axes=(1,2))), axes=(1,2))

            report = open(os.path.join(out_dir, f'symmetry_analysis_{tag}.txt'), 'w')
            report.write(f'Symmetry analysis for {pairing} pairing\n')
            report.write(f'Parameters: mu={mu}, alpha={alpha}, delta_mag={delta_mag}, gamma={gamma}, gap_size={gap_size}\n')
            report.write(f'omega={om}, v_mag={v_mag}, q0_factor={q0_factor}, Nk={Nk}, Nq={Nq}\n\n')
            report.write('C6v irrep projection (weight = sum|f_nu|^2 / sum|f|^2):\n')
            report.write(f'Datasets with max |f| < {zero_tol:.2e} are numerically zero\n')

            datasets = []
            for m, ch in enumerate(CHANNELS):
                # spatial irreps only: sigma_x, sigma_y also transform under C6v (as E1), sigma_z as A2
                sub = rf'_{ch}' if ch != '0' else ''
                for name, label, img, kind, U in [
                    (f'Re drho_{ch}(q)', rf'$\mathrm{{Re}}\,\delta\rho{sub}(\bvec{{q}})$', np.real(drho[m]), 'q', Uq),
                    (f'Im drho_{ch}(q)', rf'$\mathrm{{Im}}\,\delta\rho{sub}(\bvec{{q}})$', np.imag(drho[m]), 'q', Uq),
                    (f'drho_{ch}(r)', rf'$\delta\rho{sub}(\bvec{{r}})$', np.real(drho_r[m]), 'r', Ur),
                ]:
                    fnu, weights, maxvals = irrep_decomposition(img, geom[kind][0], U)
                    negligible = np.amax(np.abs(img)) < zero_tol
                    if negligible:
                        weights = np.zeros_like(weights)
                    datasets.append(dict(name=name, label=label, img=img, kind=kind,
                                         fnu=fnu, weights=weights, maxvals=maxvals, negligible=negligible))

                report.write(f'\nChannel sigma_{ch}:\n')
                for d in datasets[-3:]:
                    write_irrep_analysis(report, d['name'], d['weights'], d['maxvals'], d['negligible'])
                report.write(f'Check: max |Im drho_{ch}(r)| = {np.amax(np.abs(np.imag(drho_r[m]))):.4e}\n')
            report.close()

            plot_drho(datasets, geom, r_window, os.path.join(out_dir, f'drho_{tag}.png'), title)
            plot_irreps(datasets, geom, r_window, os.path.join(out_dir, f'irreps_{tag}.png'), title)


if __name__=='__main__':
    for pairing in ['s', 'ext s', 'd+id', 'dx2y2']:
        q_factors = np.array([1])
        main(pairing, q0_factors=q_factors)
