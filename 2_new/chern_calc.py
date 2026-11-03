# Chern number calculation script
import numpy as np


def setup_mesh(points, b):
    r_vals = np.arange(points) #+0.01*np.ones(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    # b1 = np.array([0,4*np.pi/np.sqrt(3)])
    # b2 = np.array([2*np.pi,2*np.pi/np.sqrt(3)])
    kx = (b[0,0]*r1 + b[1,0]*r2)/points
    ky = (1e-4+b[0,1]*r1 + b[1,1]*r2)/points
    mesh = np.swapaxes(np.array([kx.flatten(),ky.flatten()]),0,1)
    return np.reshape(mesh,(points,points,2))


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
        rows = 7

    path_to_W90 = '/home/hpc/k102de/k102de12/' +\
        'soc_materials/wannier90_files/DFT+U/' + material
    file_in = "/sym_wannier90_hr.dat"
    sys_params, initial_factors, data = read_w90_data(path_to_W90 + file_in, rows)
    factor = np.kron(initial_factors, np.ones(4)).flatten()
    ff_wan = data[:,[0,1,3,4]].astype('int')
    ff_wan[:,2:] -= 1
    n_ff = ff_wan.shape[0]
    t = (data[:,5] + 1j*data[:,6])/factor

    return ff_wan, t, sys_params


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


def calculate_mu(fil, bloch):
    E = np.linalg.eigvalsh(bloch)
    E_sorted = np.sort(E.flatten())
    en_levels = len(E.flatten())
    up_ind = int(np.ceil(fil*en_levels))
    mu_val = E_sorted[up_ind-1]
    return mu_val


def rotate_evecs(u):
    v = np.zeros_like(u)
    u0, u1 = u[0,0,3], u[1,0,3]
    v[0,:,:] = u1*u[0,:,:] - u0*u[1,:,:]
    v[1,:,:] = np.conj(u0)*u[0,:,:] + np.conj(u1)*u[1,:,:]
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
    evecs_sing = rotate_evecs(evecs_sing)
    evecs_ = np.einsum('rs,isf->irf', np.linalg.inv(spin_xfm),evecs_sing)
    evecs_k = np.reshape(np.einsum(
            'ijf,klf->iklj',
            evecs_,
            np.exp(1j*np.einsum('ijk,fk->ijf',bz,bond))
        ),(-1,bz.shape[0],bz.shape[1],2,2))

    return evecs_k


def calculate_chern_nos(eigensys):
    # eigensys = np.reshape(eigensys,(points,points,nband,nband))
    eigensys = np.swapaxes(eigensys,2,3)
    conj_eigensys = np.conjugate(eigensys)

    u1 = np.einsum(
        'rsni,rsni->rsn',
        np.roll(conj_eigensys,1,axis=0),
        eigensys
    )
    u1 = u1/np.abs(u1)
    u2 = np.einsum(
        'rsni,rsni->rsn',
        np.roll(conj_eigensys,(1,1),axis=(0,1)),
        np.roll(eigensys,1,axis=0)
    )
    u2 = u2/np.abs(u2)
    u3 = np.einsum(
        'rsni,rsni->rsn',
        np.roll(conj_eigensys,1,axis=1),
        np.roll(eigensys,(1,1),axis=(0,1))
    )
    u3 = u3/np.abs(u3)
    u4 = np.einsum(
        'rsni,rsni->rsn',
        conj_eigensys,
        np.roll(eigensys,1,axis=1)
    )
    u4 = u4/np.abs(u4)
    F = -np.log( u1 * u2 * u3 * u4 )
    chern = np.sum(np.imag(F),axis=(0,1))/(2*np.pi)
    chernnos = np.round(np.real(chern))
    # print(chernnos)

    return chernnos, F


def calculate_chern(M, O, dope, deltas_vals):
    # M = ro.read(x_dir + 'T.dvg')
    # O = ro.read(x_dir + 'O.dvg')

    lattice = M.lattice[:2,:2]
    rlattice = M.rlattice[:2,:2]
    bond = get_bond_vectors(O, lattice)[:,0,0,:]

    # nk = M.nk
    # kmesh = np.reshape(M.kmesh,(nk[0],nk[1],3))[:,:,:2]
    kmesh = setup_mesh(360, rlattice)

    ## Bloch hamiltonian
    ff_wan, t, _ = bloch_ff_from_w90('SnSi111')
    bloch = calculate_bloch(ff_wan, t, 0, kmesh, lattice)
    hole_bloch = -np.swapaxes(calculate_bloch(ff_wan, t, 0, -kmesh, lattice),2,3)

    mu = calculate_mu(0.5+dope, bloch)
    for ii in range(2):
        bloch[:,:,ii,ii] -= mu
        hole_bloch[:,:,ii,ii] += mu

    ## Superconducting instability
    evecs_k = calculate_sc_order_params(O, kmesh, bond)
    delt = (evecs_k[0,:,:,:,:] + 1j*evecs_k[1,:,:,:,:])/np.sqrt(2)

    ## BdG hamiltonian
    def bdg_ham(d):
        h = np.zeros((kmesh.shape[0],kmesh.shape[1],4,4),dtype=complex)
        h[:,:,:2,:2] = bloch
        h[:,:,2:,:2] = d*delt
        h[:,:,:2,2:] = d*np.conj(np.swapaxes(delt,2,3))
        h[:,:,2:,2:] = hole_bloch
        return h

    gap = np.zeros_like(deltas_vals)
    chern = np.zeros_like(deltas_vals)
    for ii, d in enumerate(deltas_vals):
        evals, eigensys = np.linalg.eigh(bdg_ham(d))
        gap[ii] = 2*np.amin(np.abs(evals))
        chernnos, _ = calculate_chern_nos(eigensys)
        chern[ii] = np.sum(chernnos[:2])

    return gap, chern

