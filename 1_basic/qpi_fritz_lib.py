# QPI plots calculated on Fritz
# import diverge #type:ignore
# import diverge.output as ro #type: ignore
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad #type:ignore
# from scipy.optimize import minimize #type:ignore
# from tqdm import tqdm, trange #type:ignore
# import matplotlib.pyplot as plt
import platform

## Reading in Wannier 90 files

def read_w90_data(file, r):
    data = []
    initial_factors = []
    sys_params = []
    with open(file, 'r') as f:
        d0 = f.readlines()[1:3]
        for i in d0:
            k = [i.rstrip().split(" ")[-1]]
            sys_params.append([i for i in k])
    with open(file, 'r') as f:
        d1 = f.readlines()[3:r]
        for i in d1:
            k = i.rstrip().split("    ")[1:]
            initial_factors.extend(k)
    with open(file, 'r') as f:
        d2 = f.readlines()[r:]
        for i in d2:
            k = i.rstrip('\n').split(" ")
            k[:] = [x for x in k if x]
            data.append([i for i in k])

    a = np.array(sys_params, dtype=int).flatten()
    b = np.array(initial_factors, dtype=int).flatten()
    return a, b, np.array(data, dtype=float)


def bloch_ff_from_w90(material):
    if material == "SnSi111" or material == "PbSi111":
        rows = 10
    else:
        rows = 6

    if platform.node() == '2047L-204773-M.local' or platform.node() == 'UML-2K73N8Q6LR':
        fol = '/Users/mbunney/OneDrive/Documents/PhD_Research/'
    else:
        fol = '/home/hpc/k102de/k102de12/'
    path_to_W90 = fol + 'soc_materials/wannier90_files/DFT+U/' + material
    file_in = "/sym_wannier90_hr.dat"
    sys_params, initial_factors, data = read_w90_data(path_to_W90 + file_in, rows)
    factor = np.kron(initial_factors, np.ones(4)).flatten()
    ff_wan = data[:,[0,1,3,4]].astype('int')
    ff_wan[:,2:] -= 1
    n_ff = ff_wan.shape[0]
    t = (data[:,5] + 1j*data[:,6])/factor

    return ff_wan, t, sys_params


## System setup

def get_bond_vectors(O,lattice):
    n_orbff = O.n_orbff
    n_orb = O.n_orb

    of = O.mi_to_ofrom
    ot = O.mi_to_oto
    mi_to_R = np.array(O.mi_to_R)
    mi_to_R = mi_to_R[:,:2]

    bond = np.zeros((n_orbff,n_orb,n_orb,2))
    for i in range(n_orbff):
        bond[i,of[i],ot[i],:] += np.einsum('i,ij',mi_to_R[i,:], lattice)
    return bond


def setup_mesh(points, b):
    r_vals = np.arange(points) #+0.01*np.ones(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    # b1 = np.array([0,4*np.pi/np.sqrt(3)])
    # b2 = np.array([2*np.pi,2*np.pi/np.sqrt(3)])
    kx = (b[0,0]*r1 + b[1,0]*r2)/points
    ky = (b[0,1]*r1 + b[1,1]*r2)/points
    mesh = np.swapaxes(np.array([kx.flatten(),ky.flatten()]),0,1)
    return np.reshape(mesh,(points,points,2))


## Bloch, SC OP and BdG matrix calcs

def calculate_bloch(ff_wan, t, mu, bz, latt):
    bloch = np.zeros((bz.shape[0], bz.shape[1], 2, 2), dtype=complex)
    for ii in range(ff_wan.shape[0]):
        bond = np.einsum('i,ij',ff_wan[ii,:2],latt)
        if np.linalg.norm(bond) < 1e-3:
            continue
        bloch[:,:,ff_wan[ii,2],ff_wan[ii,3]] += t[ii]*np.exp(1j*np.einsum('i,jki',bond,bz))
    for jj in range(2):
        bloch[:,:,jj,jj] += mu
    return bloch


def rotate_evecs(u):
    v = np.zeros_like(u)
    u0, u1 = u[0,0,3], u[1,0,3]
    v[0,:,:] = u1*u[0,:,:] - u0*u[1,:,:]
    v[1,:,:] = u0*u[0,:,:] + u1*u[1,:,:]
    for ii in range(v.shape[0]):
        phase = np.exp(1j*np.angle(v[ii,:,:].flatten()[np.argmax(np.abs(v[ii,:,:]))]))
        v[ii,:,:] /= np.linalg.norm(v[ii,:,:])*phase
    return v


def calculate_sc_order_params(O, bz, bond):

    n_orbff = O.n_orbff

    evecs = np.reshape(O.Pvec,(-1,4,n_orbff))
    spin_xfm = (1/2)*np.array([
            [0,1,-1,0],
            [-1,0,0,1],
            [-1j,0,0,-1j],
            [0,1,1,0]
        ])

    evecs_sing = np.einsum('rs,isf->irf',spin_xfm,evecs)
    if evecs_sing.shape[0] > 1:
        evecs_sing = rotate_evecs(evecs_sing)
    evecs_ = np.einsum('rs,isf->irf', np.linalg.inv(spin_xfm),evecs_sing)
    evecs_k = np.reshape(np.einsum(
            'ijf,klf->iklj',
            evecs_,
            np.exp(1j*np.einsum('ijk,fk->ijf',bz,bond))
        ),(-1,bz.shape[0],bz.shape[1],2,2))

    return evecs_k


def bdg_ham(bloch, hole_bloch, delt):
    # Calculate BdG Hamiltonian
    h = np.zeros((bloch.shape[0],bloch.shape[1],4,4),dtype=complex)
    h[:,:,:2,:2] = bloch
    h[:,:,2:,:2] = delt
    h[:,:,:2,2:] = np.conj(np.swapaxes(delt,2,3))
    h[:,:,2:,2:] = hole_bloch
    return h


## Green's function calculations

def calculate_greens_function(omega, gamma, en, u):
    N = u.shape[0]
    M = u.shape[-1]
    # g0 = np.empty((len(omega), N, N, M, M), complex)
    # with tqdm(total=len(omega)*N*N) as pbar:
    # for jj, kk in np.ndindex(N,N):
            # pbar.update(1)
    g0 = np.einsum( 'jkab,ojkb,jkbc->ojkac',
        u,
        1/(np.reshape(omega,(len(omega),1,1,1)) + 1j*gamma - en),
        np.conjugate(np.swapaxes(u,2,3))
    )
            # g0[:,jj,kk,:,:] = u[jj,kk,:,:] @ np.diag(1/(np.reshape(omega,(len(omega),1,1,1)) + 1j*gamma - en[jj,kk,:])) @ u[jj,kk,:,:].T.conj()
    return g0


def calculate_t_matrix(omega, v, g0r):
    t = np.empty_like(g0r)
    for ii in range(len(omega)):
        t[ii, :, :] = np.linalg.solve( np.eye(4) - v @ g0r[ii, :, :], v)
    return t


def calculate_s_matrix(t, g0r, m_g0r):
    s = np.empty_like(g0r)
    for ii in range(s.shape[0]):
        s[ii, :, :, :, :] = np.einsum(
            'abij, jk, abkl -> abil', g0r[ii,:,:,:,:], t[ii,:,:], m_g0r[ii,:,:,:,:]
        )
    return s


def spectral_function(g0k):
    return - g0k.imag / np.pi



def calculate_mu(bloch, gamma, goal_fill):

    # Diagonalize the Bloch Hamiltonian
    en, u = np.linalg.eigh(bloch)
    min_om = np.amin(en)
    ran = np.amax(en) - np.amin(en)
    buff = 0.05 * ran
    n = 0

    # Define the density of states function
    def rho0_diag(om):
        nonlocal n
        n += len(om)
        g0 = calculate_greens_function(om, gamma, en, u)
        Ak = spectral_function(g0)
        rho0 = Ak.mean(axis=(1, 2))
        return rho0[:, 0, 0] + rho0[:, 1, 1]

    # Initial range of omega values
    om_vals_0 = np.linspace(min_om - 2 * buff, 0.2, num=400, endpoint=True)
    rho0_0 = rho0_diag(om_vals_0)

    # Find the van Hove singularity
    vh_om = om_vals_0[np.argmax(rho0_0)]
    om_vals_vh = np.linspace(vh_om - 0.2 * buff, vh_om + 0.2 * buff, num=50, endpoint=True)
    om_vals_1 = om_vals_vh[~np.any(om_vals_vh[:, np.newaxis] == om_vals_0[np.newaxis, :], axis=1)]
    rho0_1 = rho0_diag(om_vals_1)

    # Combine and sort omega values
    om_vals_2 = np.sort(np.append(om_vals_0, om_vals_1).flatten())
    rho0_2 = (np.append(rho0_0, rho0_1).flatten())[np.argsort(np.append(om_vals_0, om_vals_1).flatten())]
    filling_2 = cumulative_trapezoid(rho0_2, x=om_vals_2, initial=0)

    # Refine the range around the goal filling
    rough_om = np.argmin(np.abs(filling_2 - goal_fill))
    om_vals_3 = np.linspace(om_vals_2[rough_om - 1], om_vals_2[rough_om + 1], num=400, endpoint=False)
    om_vals_3 = om_vals_3[~np.any(om_vals_3[:, np.newaxis] == om_vals_2[np.newaxis, :], axis=1)]
    rho0_3 = rho0_diag(om_vals_3)

    # Combine and sort omega values again
    om_vals_4 = np.sort(np.append(om_vals_2, om_vals_3).flatten())
    rho0_4 = (np.append(rho0_2, rho0_3).flatten())[np.argsort(np.append(om_vals_2, om_vals_3).flatten())]
    filling_4 = cumulative_trapezoid(rho0_4, x=om_vals_4, initial=0)

    # Find the chemical potential
    mu_ii = np.argmin(np.abs(filling_4 - goal_fill))
    mu = om_vals_4[mu_ii]
    fill = filling_4[mu_ii]
    # print(mu_ii, mu, fill)


    # Uncomment the following lines to check integration with Gaussian quadrature
    # # Define the density of states function for a single omega value
    # def rho0_diag_float(om):
    #     g0 = calculate_greens_function(np.array([om]), gamma, en, u)
    #     Ak = spectral_function(g0)
    #     rho0 = Ak.mean(axis=(1, 2))
    #     return rho0[0, 0, 0] + rho0[0, 1, 1]

    # # Integrate the density of states function
    # y, err, info = quad(rho0_diag_float, min_om - 2 * buff, mu, full_output=1)
    # print(y, err, info)

    # # Refine the chemical potential
    # mu_ii = np.argmin(np.abs(filling_4 - y))
    # mu = om_vals_4[mu_ii]
    # fill = filling_4[mu_ii]
    # print('\n', mu_ii, mu, fill)
    # print('\n\n', filling_4[0], rho0_4[-1])


    # Uncomment the following lines to plot the results
    # plt.figure()
    # plt.vlines([mu], 0, np.amax(rho0_4))
    # plt.plot(om_vals_4, rho0_4, '.')

    # plt.figure()
    # plt.plot(om_vals_4, filling_4, '.')
    # plt.show()

    return mu


def get_experimental_data(file_path):

    # Read the data from the file
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('%'):
                continue  # Skip header lines
            values = line.split()
            data.append([float(value) for value in values])

    data = list(zip(*data)) #type:ignore

    return data[0], data[1]
