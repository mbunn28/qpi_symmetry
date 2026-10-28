# QPI plots
import os
import sys
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
            arr[m,n] = 0
            ind += 1
            if f is not None:
                f.write(str(np.unique(indices, axis=0).shape) + ',' +str(ind) + '\n')
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


def main(pairing, om_val):

    # folder = '/Users/mbunney/Library/CloudStorage/OneDrive-Personal/Documents/PhD_Research/qpi_symmetry/'
    folder = os.path.dirname(os.path.abspath(__file__)) + '/'

    # pairing = 's'  # 'd+id', 'dx2y2', 'ext s'

    ##############
    # Defining parameters for simulation
    ##############

    mu = -0.5
    delta_mag = 1e-1
    gamma = 1e-3
    q0_factor = 2

    Nk = 1200
    Nq = 120

    sav_dir = folder + f'data_spartan_A2/D{np.abs(np.log10(delta_mag)):.2f}_G{np.abs(np.log10(gamma)):.3f}/'
    if not os.path.exists(sav_dir):
        print(f"Data directory {sav_dir} does not exist")
        return

    pairing_dir = sav_dir + pairing.replace('+','p') + '/'
    if not os.path.exists(pairing_dir):
        print(f"Pairing directory {pairing_dir} does not exist")
        return

    ##############
    # Load dispersion data
    ##############
    
    try:
        disp_data = np.load(sav_dir + 'dispersion_data.npz')
        kx = disp_data['kx']
        ky = disp_data['ky']
        disp = disp_data['disp']
        bdg_en0 = disp_data['bdg_en0']
        bdg_en1_log = disp_data['bdg_en1_log']
        rlattice = disp_data['rlattice']
    except Exception as e:
        print(f"Error loading dispersion data: {e}")
        return

    # Some plotting setup
    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    a1, a2 = lattice
    b1, b2 = rlattice

    ii, jj = np.mgrid[-int(Nq/2):int(Nq/2), -int(Nq/2):int(Nq/2)]
    rx = ii * a1[0] + jj * a2[0]
    ry = ii * a1[1] + jj * a2[1]
    qx = ii * b1[0] + jj * b2[0]
    qy = ii * b1[1] + jj * b2[1]

    # Plot dispersion
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    z = disp
    im = ax[0].pcolormesh(kx, ky, z)
    ax[0].set_title('Normal state dispersion')
    plt.colorbar(im, ax=ax[0])
    a=1.2*rlattice[1,1]
    z = bdg_en0
    im = ax[1].pcolormesh(kx, ky, z)
    ax[1].set_title('BdG spectrum')
    plt.colorbar(im, ax=ax[1])
    z = bdg_en1_log
    im = ax[2].pcolormesh(kx, ky, z)
    ax[2].set_title('Log BdG spectrum')
    plt.colorbar(im, ax=ax[2])
    for aa in range(3):
        ax[aa].set_xlim([-a,a])
        ax[aa].set_ylim([-a,a])
        ax[aa].set_aspect('equal')
    plt.savefig(sav_dir + 'dispersion_plots.png', dpi=300)
    plt.close()

    v_mag = 1e-1
    om = om_val

    ##############
    # Load QPI data
    ##############

    qpi_file = pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.npz'
    
    # If exact omega not found, list available files and pick closest
    if not os.path.exists(qpi_file):
        import glob
        available_files = glob.glob(pairing_dir + f'drho_om*_V{v_mag:.4f}_q0{q0_factor:.4f}.npz')
        if not available_files:
            print(f"No QPI data found in {pairing_dir}")
            return
        
        # Extract omega values and find closest
        import re
        omegas = []
        for f in available_files:
            match = re.search(r'drho_om([-+]?[0-9]*\.?[0-9]+)', f)
            if match:
                omegas.append((float(match.group(1)), f))
        
        if not omegas:
            print(f"Could not parse omega values from files in {pairing_dir}")
            return
        
        closest_om, qpi_file = min(omegas, key=lambda x: abs(x[0] - om))
        print(f"Omega {om:.4f} not found; using closest available: {closest_om:.4f}")
    
    try:
        qpi_data = np.load(qpi_file)
        drho = qpi_data['drho']
        drho_r = qpi_data['drho_r']
        indices_q = qpi_data['indices_q']
        indices_r = qpi_data['indices_r']
    except Exception as e:
        print(f"Error loading QPI data: {e}")
        return

    def cut_data(data):
        # cut = 1.1*np.sort(np.abs(data.flatten()))[-100]
        # data[ data > cut ] = cut
        # data[ data < -cut ] = -cut
        return data

    # Plot 2x2: real and imaginary parts in q and r space
    fig, ax = plt.subplots(2,2, figsize=(12,12))

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

            if bb == 0:
                dr = np.copy(np.real(arr))
                part = 'Re'
            else:
                dr = np.copy(np.imag(arr))
                part = 'Im'

            dr = cut_data(dr)
            z = dr

            im = ax[aa, bb].pcolormesh(ix, iy, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
            plt.colorbar(im, ax=ax[aa,bb])

            ax[aa, bb].set_title( part + r'[d$\rho (\boldsymbol{' + iname + '})$]')
            ax[aa,bb].set_facecolor('grey')
            ax[aa,bb].set_aspect('equal')


    fig.suptitle(
        pairing +\
        rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}' +\
        '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b1|$'
    )
    plt.tight_layout()
    plt.savefig(pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.png', dpi=300)
    plt.close()

    # Plot with symmetry points marked
    fig, ax = plt.subplots(1,2, figsize=(14,6))
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
            plt.colorbar(im, ax=ax[bb])

            for pp in range(indices_q.shape[0]):
                p = indices_q[pp,:]
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
        rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}' +\
        '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b1|$'
    )
    plt.tight_layout()
    plt.savefig(pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}_points.png', dpi=300)
    plt.close()

    print(f"Plots saved to {pairing_dir}")


if __name__=='__main__':
    pairings = ['s', 'ext s', 'd+id']
    om_vals = [0, 0.9]
    ps, ovs = np.meshgrid(pairings, om_vals)
    ps = ps.flatten()
    ovs = ovs.flatten()
    main(ps[int(sys.argv[1])], ovs[int(sys.argv[1])])