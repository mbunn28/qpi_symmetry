# QPI plots
import os
import numpy as np
import matplotlib.pyplot as plt

def setup_mesh(points, b):
    r_vals = np.arange(points) #+0.01*np.ones(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    # b1 = np.array([0,4*np.pi/np.sqrt(3)])
    # b2 = np.array([2*np.pi,2*np.pi/np.sqrt(3)])
    kx = (b[0,0]*r1 + b[1,0]*r2)/points
    ky = (b[0,1]*r1 + b[1,1]*r2)/points
    mesh = np.swapaxes(np.array([kx.flatten(),ky.flatten()]),0,1)
    return np.reshape(mesh,(points,points,2))


def calculate_bloch(mu, bz, latt):
    bloch = np.zeros((bz.shape[0], bz.shape[1],2,2), dtype=complex)
    disp = 2*(
        np.cos(np.einsum('i,jki',latt[0,:],bz)) +
        np.cos(np.einsum('i,jki',latt[1,:],bz)) +
        np.cos(np.einsum('i,jki',latt[0,:]+latt[1,:],bz))
    ) + mu
    bloch[:,:,0,0] = disp
    bloch[:,:,1,1] = disp
    return bloch

def calculate_sc_order_params(bz, latt, pairing):

    if pairing == 'd+id':
        d = np.array([np.exp(1j*0), np.exp(1j*2*np.pi/3), np.exp(1j*4*np.pi/3)])
    elif pairing == 'ext s':
        d = np.array([1,1,1])
    else:
        raise ValueError('Pairing not recognized')

    op = 2*(
        d[0]*np.cos(np.einsum('i,jki',latt[0,:],bz)) +
        d[1]*np.cos(np.einsum('i,jki',latt[1,:],bz)) +
        d[2]*np.cos(np.einsum('i,jki',latt[0,:]+latt[1,:],bz))
    )

    delta = np.zeros((bz.shape[0], bz.shape[1], 2,2), dtype=complex)
    delta[:,:,0,1] = op
    delta[:,:,1,0] = -op
    return delta


def main():

    folder = '/Users/mbunney/Library/CloudStorage/OneDrive-Personal/Documents/PhD_Research/qpi_symmetry/'

    pairing = 'd+id'

    ##############
    # Defining parameters for simulation
    ##############

    mu = -0.1
    delta_mag = 1e-1
    gamma = 1e-3

    N = 600

    ##############
    # Building model
    ##############

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(N, rlattice)

    ## Some plotting setup
    a1, a2 = lattice
    b1, b2 = rlattice
    center_sup = 1  #-  0.8 * np.exp( - dist**2 / (2 * 0.5**2) )

    ii, jj = np.mgrid[-N:N, -N:N]
    rx = ii * a1[0] + jj * a2[0]
    ry = ii * a1[1] + jj * a2[1]
    i, j = np.mgrid[-N:N, -N:N]
    qx = (i * b1[0] + j * b2[0])/N
    qy = (i * b1[1] + j * b2[1])/N


    ## Bloch hamiltonian
    disp = calculate_bloch(mu, kmesh, lattice)

    ## Superconducting instability
    op = calculate_sc_order_params(kmesh, lattice, pairing)

    bdg_ham = np.zeros((N,N,4,4), dtype=complex)
    bdg_ham[:,:,:2,:2] = disp
    bdg_ham[:,:,2:,2:] = -disp
    bdg_ham[:,:,:2,2:] = delta_mag * op
    bdg_ham[:,:,2:,:2] = delta_mag * np.swapaxes(np.conjugate( op ),2,3)

    m_bdg_ham = np.zeros((N,N,4,4),dtype=complex)
    m_bdg_ham[:,:,:2,:2] = disp
    m_bdg_ham[:,:,2:,2:] = -disp
    m_bdg_ham[:,:,:2,2:] = delta_mag * op
    m_bdg_ham[:,:,2:,:2] = delta_mag * np.swapaxes(np.conjugate( op ),2,3)

    sav_dir = folder + f'data/D{np.abs(np.log10(delta_mag)):.2f}_G{np.abs(np.log10(gamma)):.3f}/'
    if not os.path.exists(sav_dir):
        os.mkdir(sav_dir)

    en, u = np.linalg.eigh(bdg_ham)
    m_en, m_u = np.linalg.eigh(m_bdg_ham)
    # print(np.sum( np.abs( en) < 1e-6 ))

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    z = np.tile(np.real(disp[:,:,0,0]), (2, 2))
    im = ax[0].pcolormesh(qx, qy, z)
    ax[0].set_title('Normal state dispersion')
    plt.colorbar(im)
    a=1.2*rlattice[1,1]
    z = np.tile(en[:,:,0], (2, 2))
    im = ax[1].pcolormesh(qx, qy, z)
    ax[1].set_title('BdG spectrum')
    plt.colorbar(im)
    z = np.tile(np.log10(en[:,:,2]), (2, 2))
    im = ax[2].pcolormesh(qx, qy, z)
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

    omega_vals = np.array([-0.9*gap_size, 0, 0.9*gap_size]) #np.linspace( -0.9*gap_size, 0.9*gap_size, num= 10 )

    ## Bare Green's functions

    def calculate_greens_function(omega, gamma, en, u):
        N = u.shape[0]
        M = u.shape[-1]
        g0 = np.einsum( 'jkab,ojkb,jkbc->ojkac',
            u,
            1/(np.reshape(omega,(len(omega),1,1,1)) + 1j*gamma - en),
            np.conjugate(np.swapaxes(u,2,3))
        )
        return g0

    g0k = calculate_greens_function(omega_vals, gamma, en, u)
    m_g0k = calculate_greens_function(omega_vals, gamma, m_en, m_u)
    g0r = np.fft.ifft2(g0k, axes=(1,2))
    m_g0r = np.fft.ifft2(m_g0k, axes=(1,2))

    g0k_imp = calculate_greens_function(omega_vals, 0, en, u)
    g0r_imp = np.fft.ifft2(g0k_imp, axes=(1,2))

    v_struct = np.diag([1,1,-1,-1])

    lam_g, _ = np.linalg.eig(
        np.einsum( 'oab,bc->oac', g0r_imp[:, 0, 0, :, :], v_struct)
    )
    print(lam_g)

    v_vals = np.zeros_like( lam_g )
    drho_array = np.zeros((omega_vals.shape[0], 4, N, N))

    pairing_dir = sav_dir + pairing.replace('+','p') + '/'
    if not os.path.exists(pairing_dir):
        os.mkdir(pairing_dir)

    for ii, om in enumerate( omega_vals ):
        for jj in range(2):
            jj += 1

            if np.abs(np.imag(lam_g[ii,jj])) < 1e-6:
                v_mag = 1/( np.real(lam_g[ii,jj]))
                v_vals[ii,jj] = v_mag
                v = v_mag * v_struct
            else:
                v_mag = 1/( np.real(lam_g[ii,jj]))
                v_vals[ii,jj] = v_mag
                v = v_mag * v_struct

            ## T matrix
            T = np.linalg.solve( np.eye(4) - v @ g0r[ii, 0, 0, :, :], v)

            ## S matrix
            S = np.einsum(
                'abij, jk, abkl -> abil', g0r[ii,:,:,:,:], T, m_g0r[ii,:,:,:,:]
            )

            drho_r = -np.imag(S[:,:,0,0]+S[:,:,1,1])/np.pi
            drho = np.fft.fft2(drho_r)
            print(np.sum(np.abs(np.real(drho))),np.sum(np.abs(np.imag(drho))))
            print(np.sum(np.abs(np.real(drho_r))),np.sum(np.abs(np.imag(drho_r))))
            print(np.amax(np.abs(drho)), np.amax(np.abs(drho_r)))
            print(np.sum(np.isnan(drho_r)))
            # drho_array[ii,jj,:,:] = drho

            def cut_data(data):
                cut = 1.1*np.sort(np.abs(data.flatten()))[-100]
                data[ data > cut ] = cut
                data[ data < -cut ] = -cut
                return data

            fig, ax = plt.subplots(2,2)

            dr = np.copy(np.real(drho_r))
            dr = cut_data(dr)
            z = np.tile(dr*center_sup, (2, 2))

            im = ax[0,0].pcolormesh(rx, ry, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
            plt.colorbar(im)
            ax[0,0].set_title(r'Re[d$\rho (\boldsymbol{r})$]')

            dr = np.copy(np.imag(drho_r))
            dr = cut_data(dr)

            z = np.tile(dr*center_sup, (2, 2))

            im = ax[0,1].pcolormesh(rx, ry, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
            plt.colorbar(im)
            ax[0,1].set_title(r'Im[d$\rho (\boldsymbol{r})$]')

            dr = np.copy(np.real(drho))
            dr = cut_data(dr)

            z = np.tile(dr*center_sup, (2, 2))

            im = ax[1,0].pcolormesh(qx, qy, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
            plt.colorbar(im)
            a=1.2*rlattice[1,1]
            ax[1,0].set_xlim([-a,a])
            ax[1,0].set_ylim([-a,a])
            ax[1,0].set_title(r'Im[d$\rho (\boldsymbol{q})$]')

            dr = np.copy(np.imag(drho))
            dr = cut_data(dr)

            z = np.tile(dr*center_sup, (2, 2))

            im = ax[1,1].pcolormesh(qx, qy, z, cmap='bwr', vmin = -np.amax(np.abs(z)), vmax = np.amax(np.abs(z)))
            plt.colorbar(im)
            a=1.2*rlattice[1,1]
            ax[1,1].set_xlim([-a,a])
            ax[1,1].set_ylim([-a,a])
            ax[1,1].set_title(r'Im[d$\rho (\boldsymbol{q})$]')

            for aa, bb in np.ndindex(2,2):
                ax[aa,bb].set_facecolor('grey')


            fig.suptitle(
                pairing +\
                rf' pairing, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}' +\
                '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}'
            )
            plt.tight_layout()
            plt.savefig(pairing_dir + f'drho_om{om:.4f}_V{v_mag:.4f}.png', dpi=300)
            plt.close()

    print(v_vals)









    #     np.save(sav_dir_imp + 'abs_drho_run_detailed', np.abs(drho_array[:,:,::2,::2]))
    #     np.save(sav_dir_imp + 'abs_drho_run_sparse', np.abs(drho_array[:,:,::10,::10]))
    #     np.save(sav_dir_imp + 'v_vals', v_vals)

    # np.save(sav_dir + 'omega_vals', omega_vals)
    # np.save(sav_dir + 'gap_size', gap_size)




if __name__=='__main__':
    main()

