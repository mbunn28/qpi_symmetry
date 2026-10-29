# QPI plots
import os
import numpy as np
from tqdm import tqdm #type:ignore
import matplotlib.pyplot as plt

def setup_mesh(N, b):
    points = int(N/2)
    r_vals = np.arange(2*points)-points #+0.01*np.ones(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    # b1 = np.array([0,4*np.pi/np.sqrt(3)])
    # b2 = np.array([2*np.pi,2*np.pi/np.sqrt(3)])
    kx = (b[0,0]*r1 + b[1,0]*r2)/points/2
    ky = (b[0,1]*r1 + b[1,1]*r2)/points/2
    mesh = np.swapaxes(np.array([kx.flatten(),ky.flatten()]),0,1)
    return np.reshape(mesh,(2*points,2*points,2))

def fold_to_ws(mesh, b, tol=1e-9):
    """
    Map each point of mesh to its image in the hexagonal Wigner-Seitz cell of the
    lattice spanned by the rows of b. Points on the cell boundary are averaged over
    their equally short images, so the folded mesh keeps the full C6v symmetry.
    """
    shifts = np.array([i*b[0] + j*b[1] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    cand = mesh[..., None, :] + shifts
    d = np.sum(cand**2, axis=-1)
    w = d <= np.amin(d, axis=-1, keepdims=True) + tol
    return np.sum(cand * w[..., None], axis=-2) / np.sum(w, axis=-1)[..., None]

def build_c6v_perms(kmesh, tol=1e-6):
    """
    Permutation representation of C6v on the flattened k-mesh.

    U[g][i] is the flat index of g(k[i]), so for any array H of length N^2
    (e.g. disp.flatten()), H[U[g]] is H with k-space transformed by g.
    Elements are ordered [E, C6, C6^2, ..., C6^5, s, C6 s, ..., C6^5 s],
    where C6 is an anticlockwise rotation by pi/3 and s is the mirror ky -> -ky.
    Note H[U[g]] is the pullback k -> H(g k), so H[U[g1]][U[g2]] == H[U[g1 g2]].
    """
    N = kmesh.shape[0]
    k = kmesh.reshape(-1, 2)
    # mesh steps: axis 1 steps by b1/N, axis 0 steps by b2/N
    steps = np.array([kmesh[0,1] - kmesh[0,0], kmesh[1,0] - kmesh[0,0]])

    def perm_from_matrix(R):
        gk = k @ R.T
        # integer mesh coordinates (n1, n2) of g(k), measured from the mesh corner
        n = np.linalg.solve(steps.T, (gk - kmesh[0,0]).T).T
        n_int = np.rint(n)
        if np.amax(np.abs(n - n_int)) > tol:
            raise ValueError('Symmetry operation does not map the k-mesh onto itself')
        n_int = n_int.astype(int) % N  # fold back into the BZ by reciprocal lattice vectors
        return n_int[:,1]*N + n_int[:,0]

    th = np.pi/3
    c6 = perm_from_matrix(np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]))
    s = perm_from_matrix(np.diag([1.0, -1.0]))

    # U[g1 g2] = U[g1][U[g2]]
    U = [np.arange(N*N)]
    for _ in range(5):
        U.append(c6[U[-1]])
    U += [U[n][s] for n in range(6)]
    return np.array(U)

IRREP_NAMES = ['A1', 'A2', 'B1', 'B2', 'E1', 'E2']

def c6v_character_table():
    """
    Characters of C6v, rows ordered as IRREP_NAMES, columns in the element order
    of build_c6v_perms: [E, C6, C3, C2, C3^2, C6^5, s, C6 s, ..., C6^5 s].
    B1 is +1 on the mirror class of s (ky -> -ky), e.g. sin(k.a1) + sin(k.a2) + sin(k.a3)
    with the a_i at 120 degrees; B2 is -1 on it.
    """
    return np.array([
        [1,  1,  1,  1,  1,  1,    1,  1,  1,  1,  1,  1],
        [1,  1,  1,  1,  1,  1,   -1, -1, -1, -1, -1, -1],
        [1, -1,  1, -1,  1, -1,    1, -1,  1, -1,  1, -1],
        [1, -1,  1, -1,  1, -1,   -1,  1, -1,  1, -1,  1],
        [2,  1, -1, -2, -1,  1,    0,  0,  0,  0,  0,  0],
        [2, -1, -1,  2, -1, -1,    0,  0,  0,  0,  0,  0],
    ])

def project_to_irreps(H, kmesh, U=None):
    """
    Project H onto each irrep of C6v: H_nu = (d_nu/|G|) sum_g chi_nu(g) H[U[g]].

    H has shape (N^2, ...) or (N, N, ...) for an N x N kmesh. Returns Hnu with shape
    (6, *H.shape), irreps ordered as IRREP_NAMES, and Hnu.sum(axis=0) == H.
    Only the k-dependence is transformed, so a matrix-valued H (e.g. bdg_ham) is
    projected element by element. Pass U from build_c6v_perms to avoid rebuilding it.
    """
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
    # characters are real, so chi* = chi, and H[U[g]] acting as g^-1 gives the same projector
    coef = chi * chi[:, :1] / len(U)

    Hnu = 0
    for g in range(len(U)):
        Hnu = Hnu + np.multiply.outer(coef[:, g], Hf[U[g]])
    return Hnu.reshape(len(chi), *H.shape)

def irrep_decomposition(f, mesh, U):
    """
    Project a 2D image f on mesh onto the C6v irreps. Returns the components fnu,
    the weight fractions sum|f_nu|^2 / sum|f|^2 (these sum to 1) and max |f_nu|.
    """
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

def plot_irrep_components(datasets, path, suptitle, min_weight=1e-4):
    """
    datasets: list of (label, image, fnu, weights, mesh). One row per dataset: the full
    image followed by each irrep component whose weight fraction exceeds min_weight.
    """
    keep = [np.flatnonzero(weights > min_weight) for _, _, _, weights, _ in datasets]
    ncol = 1 + max(len(k) for k in keep)
    fig, ax = plt.subplots(len(datasets), ncol, figsize=(3*ncol, 3*len(datasets)), squeeze=False)

    for row, ((label, img, fnu, weights, mesh), kk) in enumerate(zip(datasets, keep)):
        panels = [(img, label)] + [(fnu[nu], f'{IRREP_NAMES[nu]}: {100*weights[nu]:.2f}%') for nu in kk]
        for col in range(ncol):
            if col >= len(panels):
                ax[row, col].axis('off')
                continue
            z, title = panels[col]
            vmax = np.amax(np.abs(z))
            im = ax[row, col].pcolormesh(mesh[:,:,0], mesh[:,:,1], z, cmap='bwr', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax[row, col])
            ax[row, col].set_title(title)
            ax[row, col].set_facecolor('grey')
            ax[row, col].set_aspect('equal')

    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def calculate_bloch(mu, bz, latt):
    disp = 2*(
        np.cos(np.einsum('i,jki',latt[0,:],bz)) +
        np.cos(np.einsum('i,jki',latt[1,:],bz)) +
        np.cos(np.einsum('i,jki',latt[0,:]+latt[1,:],bz))
    ) + mu
    return disp

def calculate_sc_order_params(bz, latt, pairing):

    if pairing == 'd+id':
        d = np.array([np.exp(1j*0), np.exp(1j*2*np.pi/3), np.exp(1j*4*np.pi/3)])
    elif pairing == 'dx2y2':
        d = np.array([1, -0.5, -0.5])
    elif pairing == 'ext s':
        d = np.array([1,1,1])
    elif pairing == 's':
        return np.ones_like(bz[:,:,0])
    else:
        raise ValueError('Pairing not recognized')

    op = 2*(
        d[0]*np.cos(np.einsum('i,jki',latt[0,:],bz)) +
        d[1]*np.cos(np.einsum('i,jki',latt[1,:],bz)) +
        d[2]*np.cos(np.einsum('i,jki',latt[0,:]+latt[1,:],bz))
    )

    return op

def calculate_bdg(disp, op):
    N = disp.shape[0]
    bdg_ham = np.zeros((N,N,2,2), dtype=complex)
    bdg_ham[:,:,0,0] = disp
    bdg_ham[:,:,1,1] = -disp
    bdg_ham[:,:,0,1] = op
    bdg_ham[:,:,1,0] = np.conjugate(op)
    return bdg_ham

def basis_functions():
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 0, 1],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 1, 1, 0],
        [1, 0, -1, -1, 0, 1, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, -1, 1, 0, -1],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 1, -1, 0],
        [1, 0, -1, 1, 0, -1, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    ])

def symmetry_analyse(arr0, f=None):

    # ind = np.argsort(np.abs(arr0.flatten()))[-1]
    # max_val = np.abs(arr0.flatten()[ind])
    ind = 0
    N = arr0.shape[0]
    arr = np.abs( np.roll(np.roll(arr0, int(N/2), axis=0), int(N/2), axis=1) )
    while True:
        m, n = np.argwhere( arr == np.amax(np.abs(arr)) )[0]
        # print(m,n, arr[m,n], np.amax(np.abs(arr0)))
        # input()
        l = (m+n)%N
        origin = int(N/2)
        indices = np.array([
            [n,m],
            [-m,l],
            [-l,n],
            [-n,-m],
            [m,-l],
            [l,-n],
            [n,-l],
            [-m,-n],
            [-l,m],
            [-n,l],
            [m,n],
            [l,-m]
        ]) + origin
        if np.unique(indices, axis=0).shape[0] == 12:
            break
        else:
            print(np.unique(indices, axis=0).shape)
            arr[m,n] = 0
            ind += 1
            if f is not None:
                f.write(str(ind) + '\n')
    if f is not None:
        f.write(f'found after {ind} tries\n')

    I_vals = np.zeros(12, dtype=complex)
    for i in range(12):
        p = indices[i,:]
        I_vals[i] = arr0[p[0]%N,p[1]%N]

    sym_I = np.dot(basis_functions(), I_vals)
    if f is not None:
        f.write("Symmetry decomp\n")
        f.write(str(sym_I) + '\n')
    # input()
    return indices


def main(pairing):

    folder = '/Users/mtb31/Library/CloudStorage/OneDrive-Personal/Documents/PhD_Research/qpi_symmetry/'

    # pairing = 's'  # 'd+id', 'dx2y2', 'ext s'

    ##############
    # Defining parameters for simulation
    ##############

    mu = -0.1
    delta_mag = 1e-1
    gamma = 1e-2
    q0_factor = 20

    Nk = 120
    Nq = 24

    sav_dir = folder + f'data/A2/D{np.abs(np.log10(delta_mag)):.2f}_G{np.abs(np.log10(gamma)):.3f}/'
    if not os.path.exists(sav_dir):
        os.mkdir(sav_dir)

    pairing_dir = sav_dir + pairing.replace('+','p') + '/'
    if not os.path.exists(pairing_dir):
        os.mkdir(pairing_dir)

    ##############
    # Building model
    ##############

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(Nk, rlattice)
    kx, ky = kmesh[:,:,0], kmesh[:,:,1]

    ## q-mesh over the BZ and its Fourier-conjugate r-mesh of lattice sites
    # q = (m1 b1 + m2 b2)/Nq and r = n1 a1 + n2 a2, both centred, so that
    # drho(r) = fftshift(ifft2(ifftshift(drho(q)))) = (1/Nq^2) sum_q exp(i q.r) drho(q)
    if Nk % Nq != 0:
        raise ValueError('Nk must be a multiple of Nq so k+q lies on the k-mesh')
    qstep = Nk // Nq
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    qx, qy = qmesh[:,:,0], qmesh[:,:,1]
    rx, ry = rmesh[:,:,0], rmesh[:,:,1]
    Uq = build_c6v_perms(qmesh)
    Ur = build_c6v_perms(rmesh)

    # V is not periodic in k or q, so evaluate it on the symmetric (Wigner-Seitz) cell
    kws = fold_to_ws(kmesh, rlattice)
    qws = fold_to_ws(qmesh, rlattice)


    ## BdG Hamiltonian
    disp = calculate_bloch(mu, kmesh, lattice)
    op = calculate_sc_order_params(kmesh, lattice, pairing)
    bdg_ham = calculate_bdg(disp, op)
    en, u = np.linalg.eigh(bdg_ham)

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    z = np.real(disp)
    im = ax[0].pcolormesh(kx, ky, z)
    ax[0].set_title('Normal state dispersion')
    plt.colorbar(im)
    a=1.2*rlattice[1,1]
    z = np.real(en[:,:,0])
    im = ax[1].pcolormesh(kx, ky, z)
    ax[1].set_title('BdG spectrum')
    plt.colorbar(im)
    z = np.log10(en[:,:,1])
    im = ax[2].pcolormesh(kx, ky, z)
    ax[2].set_title('Log BdG spectrum')
    plt.colorbar(im)
    for aa in range(3):
        ax[aa].set_xlim([-a,a])
        ax[aa].set_ylim([-a,a])
        ax[aa].set_aspect('equal')
    plt.savefig(sav_dir + 'dispersion_plots.png', dpi=300)
    plt.close()

    gap_size = np.amin(np.abs(en))
    print(f'Gap size: {gap_size}')

    if gap_size < 1e-6:
        with open(sav_dir + 'no_gap.txt', 'w') as f:
            f.write(f'No gap found for mu {mu}, gamma {gamma}, mag_delt {delta_mag}\n')
        return

    omega_vals = np.array([0, 0.98*gap_size]) #np.linspace( -0.9*gap_size, 0.9*gap_size, num= 10 )

    ## Bare Green's functions

    def calculate_greens_function(omega, gamma, en, u):
        g0 = np.einsum( 'jkab,ojkb,jkbc->ojkac',
            u,
            1/(np.reshape(omega,(len(omega),1,1,1)) + 1j*gamma - en),
            np.conjugate(np.swapaxes(u,2,3))
        )
        return g0

    g0k = calculate_greens_function(omega_vals, gamma, en, u)

    v_mag = 1e-1
    q0 = np.linalg.norm(rlattice[0,:])*q0_factor

    for ii, om in enumerate( omega_vals ):

        S = np.zeros((Nq,Nq,2,2), dtype=complex)
        for i, j in tqdm(np.ndindex(Nq, Nq), total=Nq*Nq, desc='QPI loop'):
            # S(q) = sum_k G(k) T(k, k-q) G(k-q)
            # axis 0 of the meshes steps along b2, axis 1 along b1; rolling by +q gives k-q
            m2, m1 = i - Nq//2, j - Nq//2
            kmqws = np.roll( kws, (m2*qstep, m1*qstep), axis=(0,1) )
            qxi, qyi = qws[i,j]
            # V(k, k') = i v k x k' with both momenta folded separately; folding q instead of k-q
            # would break V(k-q, k) = V(k, k-q)^* wherever k-q crosses the zone boundary
            V = v_mag * 1j * ( kws[:,:,0] * kmqws[:,:,1] - kws[:,:,1] * kmqws[:,:,0] ) *\
                    np.exp(- (qxi**2 + qyi**2) / (q0**2) )
            T = V.reshape(Nk,Nk,1,1) * np.diag([1, -1]).reshape(1,1,2,2)

            g0kmq = np.roll( g0k[ii,:,:,:,:], (m2*qstep, m1*qstep), axis=(0,1) )
            S[i,j,:,:] = np.einsum(
            'ijrs, ijst, ijtu -> ru', g0k[ii,:,:,:,:], T, g0kmq
            )

        # FT-LDOS drho(q) = sum_r exp(-iq.r) drho(r), with drho(r) = -Im dG_00(r,r)/pi:
        #   drho(q) = -[S(q) - S(-q)^dagger]_00 / (2 pi i)    (see notes/drho_derivation.tex)
        # S is periodic on the q-mesh (V uses folded momenta), so S(-q) is just a reindexing
        neg = (Nq - np.arange(Nq)) % Nq   # index of -q on the centred q-mesh
        S_neg = S[np.ix_(neg, neg)]
        drho = - ( S[:,:,0,0] - np.conj(S_neg[:,:,0,0]) ) / (2j*np.pi)
        # drho_r = -np.imag(S[:,:,0,0]+S[:,:,1,1])/np.pi
        drho_r = np.fft.fftshift( np.fft.ifft2( np.fft.ifftshift(drho) ) )

        # |drho(q)| <= max|V| sum_k |G(k)|^2, so anything far below this bound is rounding noise
        drho_bound = v_mag * np.amax(np.sum(kws**2, axis=-1)) * np.sum(np.abs(g0k[ii])**2)
        zero_tol = 1e-10 * drho_bound

        # drho(r) is real by construction (drho(-q) = drho(q)^*), so Im drho(r) is only a sanity check
        datasets = []
        for label, name, img, mesh, U in [
            (r'Re[d$\rho (\boldsymbol{q})$]', 'Re drho(q)', np.real(drho), qmesh, Uq),
            (r'Im[d$\rho (\boldsymbol{q})$]', 'Im drho(q)', np.imag(drho), qmesh, Uq),
            (r'd$\rho (\boldsymbol{r})$', 'drho(r)', np.real(drho_r), rmesh, Ur),
        ]:
            fnu, weights, maxvals = irrep_decomposition(img, mesh, U)
            negligible = np.amax(np.abs(img)) < zero_tol
            if negligible:
                weights = np.zeros_like(weights)
                label += '\n(numerically zero)'
            datasets.append((label, img, fnu, weights, mesh, name, maxvals, negligible))

        with open(pairing_dir + f'symmetry_analysis_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.txt', 'w') as f:
            f.write(f'Symmetry analysis for {pairing} pairing\n')
            f.write(f'Parameters: delta_mag={delta_mag}, gamma={gamma}, gap_size={gap_size}\n')
            f.write(f'omega={om}, v_mag={v_mag}, q0_factor={q0_factor}\n\n')

            f.write('C6v irrep projection (weight = sum|f_nu|^2 / sum|f|^2):\n')
            f.write(f'Datasets with max |f| < {zero_tol:.2e} (1e-10 of the bound on |drho|) are numerically zero\n')
            for d in datasets:
                write_irrep_analysis(f, d[5], d[3], d[6], d[7])
            f.write(f'Check: max |Im drho(r)| = {np.amax(np.abs(np.imag(drho_r))):.4e} '
                    f'(max |drho(r)| = {np.amax(np.abs(drho_r)):.4e}); should be at rounding level\n')
            f.write('\n')

            f.write('Fourier space analysis:\n')
            indices = symmetry_analyse(drho, f=f)
            f.write('\nReal space analysis:\n')
            symmetry_analyse(drho_r, f=f)

            # f.write('\n\n')
            # f.write(str(np.sum(np.abs(np.real(drho))),np.sum(np.abs(np.imag(drho)))))
            # f.write(str(np.sum(np.abs(np.real(drho_r))),np.sum(np.abs(np.imag(drho_r)))))
            # f.write(str(np.amax(np.abs(drho)), np.amax(np.abs(drho_r))))
            # f.write(str(np.sum(np.isnan(drho_r))))
            f.close()
        # drho_array[ii,jj,:,:] = drho

        def cut_data(data):
            # cut = 1.1*np.sort(np.abs(data.flatten()))[-100]
            # data[ data > cut ] = cut
            # data[ data < -cut ] = -cut
            return data

        fig, ax = plt.subplots(2,2)

        for aa in range(2):
            for bb in range(2):

                if aa == 0:
                    arr = drho_r
                    ix, iy = rx, ry
                    iname = 'r'
                else:
                    arr = drho
                    ix, iy = qx, qy
                    iname = 'q'

                if aa == 0 and bb == 1:
                    # drho(r) is real; Im drho(r) is reported as a check in the text file
                    ax[aa, bb].axis('off')
                    continue

                if bb == 0:
                    dr = np.copy(np.real(arr))
                    part = 'Re'
                else:
                    dr = np.copy(np.imag(arr))
                    part = 'Im'

                dr = cut_data(dr)
                z = dr

                im = ax[aa, bb].pcolormesh(ix, iy, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
                plt.colorbar(im)

                ax[aa, bb].set_title( part + r'[d$\rho (\boldsymbol{' + iname + '})$]')
                ax[aa,bb].set_facecolor('grey')
                ax[aa,bb].set_aspect('equal')


        fig.suptitle(
            pairing +\
            rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}' +\
            '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b1|$'
        )
        plt.tight_layout()
        plt.savefig(pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.png', dpi=300)
        plt.close()

        plot_irrep_components(
            [d[:5] for d in datasets],
            pairing_dir + f'irreps_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.png',
            pairing +\
            rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}' +\
            '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b1|$'
        )

        fig, ax = plt.subplots(1,2)
        for aa in range(1):
            aa += 1
            for bb in range(2):

                if aa == 0:
                    arr = drho_r
                    ix, iy = rx, ry
                    iname = 'r'
                else:
                    arr = drho
                    ix, iy = qx, qy
                    iname = 'q'

                if bb == 0:
                    dr = np.copy(np.real(arr))
                    part = 'Re'
                else:
                    dr = np.copy(np.imag(arr))
                    part = 'Im'

                dr = cut_data(dr)
                z = dr

                im = ax[bb].pcolormesh(ix, iy, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
                a = 0.25*rlattice[1,1]
                # ax[bb].set_xlim([-a,a])
                # ax[bb].set_ylim([-a,a])
                plt.colorbar(im)

                for pp in range(indices.shape[0]):
                    p = indices[pp,:]
                    ax[bb].plot(
                        qx[p[0]%Nq,p[1]%Nq],
                        qy[p[0]%Nq,p[1]%Nq],
                        'ko', markersize=5, markerfacecolor='none', markeredgewidth=0.5
                    )

                ax[bb].set_title( part + r'[d$\rho (\boldsymbol{' + iname + '})$]')
                ax[bb].set_facecolor('grey')
                ax[bb].set_aspect('equal')


        fig.suptitle(
            pairing +\
            rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}' +\
            '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b1|$'
        )
        plt.tight_layout()
        plt.savefig(pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}_points.png', dpi=300)
        plt.close()












    #     np.save(sav_dir_imp + 'abs_drho_run_detailed', np.abs(drho_array[:,:,::2,::2]))
    #     np.save(sav_dir_imp + 'abs_drho_run_sparse', np.abs(drho_array[:,:,::10,::10]))
    #     np.save(sav_dir_imp + 'v_vals', v_vals)

    # np.save(sav_dir + 'omega_vals', omega_vals)
    # np.save(sav_dir + 'gap_size', gap_size)




if __name__=='__main__':
    pairings = ['s', 'ext s', 'd+id']
    for pairing in pairings:
        main(pairing)