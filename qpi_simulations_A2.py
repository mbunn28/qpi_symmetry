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
    gamma = 1e-4
    q0_factor = 2

    Nk = 600
    Nq = 24

    sav_dir = folder + f'data/D{np.abs(np.log10(delta_mag)):.2f}_G{np.abs(np.log10(gamma)):.3f}/'
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

    ## Some plotting setup
    a1, a2 = lattice
    b1, b2 = rlattice
    center_sup = 1  #-  0.8 * np.exp( - dist**2 / (2 * 0.5**2) )

    ii, jj = np.mgrid[-int(Nq/2):int(Nq/2), -int(Nq/2):int(Nq/2)]
    rx = ii * a1[0] + jj * a2[0]
    ry = ii * a1[1] + jj * a2[1]
    qx = ii * b1[0] + jj * b2[0]
    qy = ii * b1[1] + jj * b2[1]


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

    omega_vals = np.array([0, 0.9*gap_size]) #np.linspace( -0.9*gap_size, 0.9*gap_size, num= 10 )

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
        for i_j in tqdm(np.ndindex(Nq, Nq), total=Nq*Nq, desc='QPI loop'):
            i, j = i_j
            qxi, qyi = qx[i,j], qy[i,j]
            V = v_mag * 1j * ( kmesh[:,:,0] * ( kmesh[:,:,1] + qyi ) -\
                       kmesh[:,:,1] * ( kmesh[:,:,0] + qxi ) ) *\
                    np.exp(- (qxi**2 + qyi**2) / (q0**2) )
            T = V.reshape(Nk,Nk,1,1) * np.diag([1, -1]).reshape(1,1,2,2)

            g0kq = np.roll( np.roll( g0k[ii,:,:,:,:], -i, axis=0), -j, axis=1 )
            S[i,j,:,:] = np.einsum(
            'ijrs, ijst, ijtu -> ru', g0k[ii,:,:,:,:], T, g0kq
            )

        drho = S[:,:,0,0]
        # drho_r = -np.imag(S[:,:,0,0]+S[:,:,1,1])/np.pi
        drho_zero = np.roll( np.roll( drho, int(Nq/2), axis=0), int(Nq/2), axis=1 )
        drho_r_zero = np.fft.ifft2(drho_zero)
        # drho_r = np.roll( np.roll( drho_r_zero, int(Nq/2), axis=0), int(Nq/2), axis=1 )

        # drho_r_zero = np.imag( drho_r_zero )
        drho_r = np.roll( np.roll( drho_r_zero, int(Nq/2), axis=0), int(Nq/2), axis=1 )
        drho_zero = np.fft.fft2(drho_r_zero)
        drho = np.roll( np.roll( drho_zero, int(Nq/2), axis=0), int(Nq/2), axis=1 )

        with open(pairing_dir + f'symmetry_analysis_om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}.txt', 'w') as f:
            f.write(f'Symmetry analysis for {pairing} pairing\n')
            f.write(f'Parameters: delta_mag={delta_mag}, gamma={gamma}, gap_size={gap_size}\n')
            f.write(f'omega={om}, v_mag={v_mag}, q0_factor={q0_factor}\n\n')

            f.write('Fourier space analysis:\n')
            indices = symmetry_analyse(drho, f=f)
            f.write('\nReal space analysis:\n')
            indices = symmetry_analyse(drho_r, f=f)

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