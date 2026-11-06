# Plotting script for analysis SC states
import diverge.output as ro #type: ignore
import numpy as np


def get_projection_ops(output_folder, n_ff_cut=None):

    M = ro.read(output_folder + 'T.dvg')
    n_ff = M.len_ff_struct
    n_spin = M.n_spin

    O = ro.read(output_folder + 'O.dvg')
    idx_ibz_in_bz = O.idx_ibz_in_bz
    n_ipz = np.shape(idx_ibz_in_bz)[0]

    sym_maps = np.reshape(O.symm_o2m_idx_map, (-1,n_ff*(n_spin)**2,n_ipz))[:,:,0]
    n_sym = sym_maps.shape[0]
    sym_mats = np.zeros((n_sym,n_ff*(n_spin)**2,n_ff*(n_spin)**2), dtype=complex)
    P_prefac = np.reshape(O.symm_o2m_Ppref, (n_sym,n_ff*(n_spin)**2,n_ipz))[:,:,0]
    for s in range(n_sym):
        for i in range(n_ff*(n_spin)**2):
            sym_mats[s,i,sym_maps[s,i]] = P_prefac[s,i]

    character_table = np.array([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1],
        [1,-1,1,-1,1,-1,1,1,1,-1,-1,-1],
        [1,-1,1,-1,1,-1,-1,-1,-1,1,1,1],
        [2,1,-1,-2,-1,1,0,0,0,0,0,0],
        [2,-1,-1,2,-1,-1,0,0,0,0,0,0]
    ])

    proj_ops = np.einsum(
        'm,mi,ijk->mjk', character_table[:,0]/12, character_table, sym_mats[:12,:,:]
        )

    if n_spin == 2: #transform to SC singlet/triplet
        proj_ops_ = np.reshape(proj_ops, (-1, 4, n_ff, 4, n_ff))
        proj_ops = np.reshape(np.swapaxes(proj_ops_, 1, 3), (-1, 4*n_ff, 4*n_ff))
        spin_xfm = (1/np.sqrt(2))*np.array([
            [0,1,-1,0],
            [-1,0,0,1],
            [-1j,0,0,-1j],
            [0,1,1,0]
        ])
        n_ff = int(proj_ops.shape[1]/n_spin**2)
        proj_ops = np.einsum(
            'rs,tu,nsftg->nrfug', spin_xfm, np.linalg.inv(spin_xfm),
            np.reshape(proj_ops,(-1,4,n_ff,4,n_ff))
        )
        if n_ff_cut is None:
            return np.reshape(proj_ops, (-1, 4*n_ff, 4*n_ff))
        else:
            return np.reshape(proj_ops[:,:,:n_ff_cut,:,:n_ff_cut],
                                  (-1, 4*n_ff_cut, 4*n_ff_cut))

    if n_ff_cut is None:
        return proj_ops
    else:
        return proj_ops[:,:n_ff_cut,:n_ff_cut]

def symmetry_map(proj_ops, dist, n_spin, spatial_proj=None, bonds=None):

    def sort_states(states, spat_proj, xy_ind):
        n_states = states.shape[0]
        n_irrep = spat_proj.shape[0]
        n_ff = spat_proj.shape[-1]

        states = np.reshape(states, (n_states, 4, -1))
        basis_states = np.zeros( (n_states, 4, states.shape[-1]), dtype=complex)
        basis_spatial_irrep = -1*np.ones(n_states, dtype=int)

        inv_sym_spatial_irreps = [0,1,5]
        n_basis = 0
        for nn in range(n_irrep):

            ## project to spatial irrep (given fermionic AS)

            if nn in inv_sym_spatial_irreps:
                antisym_states= np.einsum(
                    'fg,ig->if',spat_proj[nn,:,:], states[:,0,:]
                )
            else:
                antisym_states = np.einsum(
                'fg,isg->isf',spat_proj[nn,:,:], states[:,1:,:]
            )

            ## Remove states w no spectral weight

            antisym_states_ = np.transpose(np.reshape(antisym_states,(n_states,-1)))
            antisym_states_norm = np.linalg.norm(antisym_states_, axis=0)
            states_w_overlap = antisym_states_norm > 1e-6
            if np.sum(states_w_overlap) == 0:
                continue
            antisym_states = antisym_states_[:, antisym_states_norm > 1e-6]

            ## Gram-schmidt orthogonalisation (via QR decomp)

            orthogonal_antisym, r = np.linalg.qr(antisym_states)
            orth_state_index = np.diag(np.abs(r)) > 1e-10
            n_new_basis = np.sum(orth_state_index)
            if n_new_basis == 0:
                continue
            antisym_states = orthogonal_antisym[ :, orth_state_index]

            ## align doubly deg states along x/y vecs

            if n_new_basis == 2:
                antisym_states_ = np.copy(antisym_states)
                if nn in inv_sym_spatial_irreps:
                    a, b = antisym_states_[xy_ind,0], antisym_states_[xy_ind,1]
                else:
                    spin_sum = np.sum(np.abs(np.reshape(antisym_states[:,0],(3,-1))),axis=1)
                    spin_ind = np.argmax(spin_sum)
                    a,b = antisym_states_[spin_ind*n_ff + xy_ind,0], antisym_states_[spin_ind*n_ff + xy_ind,1]
                antisym_states[:,0] = a*antisym_states_[:,0] + b*antisym_states_[:,1]
                antisym_states[:,1] = b*antisym_states_[:,0] - a*antisym_states_[:,1]
                for mm in range(2):
                    antisym_states[:,mm] /= np.linalg.norm(antisym_states[:,mm])


            ## write basis states to array

            if nn in inv_sym_spatial_irreps:
                basis_states[n_basis:n_basis+n_new_basis, 0, :] =\
                      np.transpose(antisym_states)
            else:
                basis_states[n_basis:n_basis+n_new_basis, 1:, :] =\
                    np.reshape(
                        np.transpose(antisym_states),
                        (-1,3,n_ff)
                    )
            basis_spatial_irrep[n_basis:n_basis+n_new_basis] = nn*np.ones(n_new_basis)
            n_basis += n_new_basis

        return np.reshape(basis_states[:n_basis,:,:],(-1,4*n_ff)), basis_spatial_irrep[:n_basis]


    # Calculate distance/shell data
    shell_dist = np.unique(np.round(dist,8))
    spin_dist = np.array([dist]*int(n_spin**2)).flatten()

    # System paramters/sizes
    n_irrep = proj_ops.shape[0]
    n_orbff = proj_ops.shape[-1]
    n_ff = spatial_proj.shape[-1]

    basis_states_total = np.zeros((n_irrep,n_orbff,n_ff*n_spin**2), dtype=complex)
    basis_shell = -1*np.ones((n_irrep,n_orbff), dtype=int)
    if n_spin == 2:
        basis_state_spat_irrep = -1*np.ones((n_irrep,n_orbff),dtype=int)
    no_states = np.zeros(n_irrep, dtype=int)

    for ii, dd in enumerate(shell_dist):

        ## Extract proj op on shell, diagonalise

        shell_mask = np.isclose(spin_dist, dd)
        shell_proj = proj_ops[:,shell_mask,:][:,:,shell_mask]
        proj_evals, proj_evecs = np.linalg.eigh(shell_proj)

        for nn in range(n_irrep):

            ## Basis states on shell for each irrep

            if np.sum(np.isclose(proj_evals[nn,:],1)) == 0:
                continue
            basis_states = proj_evecs[nn,:,np.isclose(proj_evals[nn,:],1)]

            ## If spinful, keep only fermionic AS states, and sort by spatial irrep

            if n_spin == 2:
                shell_spat_mask = np.isclose(dist, dd)
                shell_spatial_proj = spatial_proj[:,shell_spat_mask,:][:,:,shell_spat_mask]
                shell_bonds = bonds[shell_spat_mask,:]
                xy_ind = np.argmin(
                    np.abs(np.sqrt(np.sum(shell_bonds**2,axis=1)) -\
                    np.sum(np.abs(shell_bonds),axis=1))
                           )
                basis_states, basis_spatial_irrep = sort_states(basis_states, shell_spatial_proj, xy_ind)

            ## Write basis states to array

            no_bs = basis_states.shape[0]
            basis_shell[nn,no_states[nn]:no_states[nn]+no_bs] = ii*np.ones(no_bs)
            basis_states_total[nn,no_states[nn]:no_states[nn]+no_bs,:][:,shell_mask] = basis_states
            if n_spin == 2:
                basis_state_spat_irrep[nn,no_states[nn]:no_states[nn]+no_bs] = basis_spatial_irrep

            no_states[nn] += no_bs

    ## Remove phase of estates

    for ii in range(n_irrep):
        for jj in range(no_states[ii]):
            phase = np.angle(basis_states_total[ii,jj,:][np.argmax(np.abs(basis_states_total[ii,jj,:]))])
            basis_states_total[ii,jj,:] /= np.exp(1j*phase)
            if basis_state_spat_irrep[ii,jj] in [2,3,4]:
                basis_states_total[ii,jj,:] *= 1j

    max_no_states = np.amax(no_states)

    a = basis_states_total[:,:max_no_states,:]
    b = basis_shell[:,:max_no_states]
    if n_spin == 2:
        return a, b, basis_state_spat_irrep[:,:max_no_states]
    else:
        return a, b



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

def rotate_evecs(u):
    v = np.zeros_like(u)
    u0, u1 = u[0,0,3], u[1,0,3]
    v[0,:,:] = u1*u[0,:,:] - u0*u[1,:,:]
    v[1,:,:] = u0*u[0,:,:] + u1*u[1,:,:]
    for ii in range(v.shape[0]):
        phase = np.exp(1j*np.angle(v[ii,:,:].flatten()[np.argmax(np.abs(v[ii,:,:]))]))
        v[ii,:,:] /= np.linalg.norm(v[ii,:,:])*phase
    return v


def get_symmetry_map(folder, sym_maps_folder):

    M = ro.read(folder + 'T.dvg')
    n_ff = M.len_ff_struct
    lattice = M.lattice[:2,:2]

    O = ro.read(folder + 'O.dvg')
    bond = get_bond_vectors(O, lattice)[:,0,0,:]
    dist = np.sqrt(np.sum(bond**2, axis=1))

    spatial_proj_ops = get_projection_ops(sym_maps_folder, n_ff_cut = n_ff)
    proj_ops = get_projection_ops(sym_maps_folder + '/soc/', n_ff_cut = n_ff)
    sym_maps, map_shell, map_spat_irrep = symmetry_map(proj_ops, dist, 2, spatial_proj_ops, bond)

    return sym_maps, map_shell, map_spat_irrep


def analyse_sc_state(x_dir):

    M = ro.read(x_dir + 'T.dvg')
    nk = M.nk
    kmesh = np.reshape(M.kmesh,(nk[0]*nk[1]*nk[2],3))
    lattice = M.lattice[:2,:2]

    O = ro.read(x_dir + 'O.dvg')
    idx_ibz_in_bz = O.idx_ibz_in_bz
    ipz = kmesh[idx_ibz_in_bz,:]
    n_orbff = O.n_orbff
    mi_to_R = np.array(O.mi_to_R)[:,:2]

    bond = get_bond_vectors(O, lattice)[:,0,0,:]
    sym_maps, map_shell, map_spat_irrep = get_symmetry_map(x_dir)


    evals = O.Pval
    evecs = O.Pvec
    lens = O.Plen
    offs = O.Poff
    print('evals', evals)
    print('no evals total:', np.sum(lens), 'no evals at q=0:', lens[0])

    q_ind = np.zeros(len(evals),dtype=int)
    for ii, of in enumerate(offs):
        q_ind[of:of+lens[ii]] = int(ii)

    max_eval = np.amax(evals)
    max_evecs_inds = np.argwhere(np.isclose(np.abs(evals),max_eval))

    evecs = np.reshape(evecs[max_evecs_inds],(-1,4,n_orbff))
    evals = evals[max_evecs_inds]
    q_ind = q_ind[max_evecs_inds]

    evecs = np.reshape(evecs,(-1,4,n_orbff))
    spin_xfm = (1/2)*np.array([
            [0,1,-1,0],
            [-1,0,0,1],
            [-1j,0,0,-1j],
            [0,1,1,0]
        ])

    evecs_sing = np.einsum('rs,isf->irf',spin_xfm,evecs)
    evecs_sing = rotate_evecs(evecs_sing)
    evecs_sing = np.reshape(evecs_sing,(-1,4*n_orbff))
    irreps = np.einsum('nif,jf->jni', sym_maps, evecs_sing)

    for n_vec in range(len(evals)):

        print('eval:', evals[n_vec])
        print('q:', ipz[q_ind[n_vec],:])
        for bb in range(4):
            print(bb)
            for aa in range(n_orbff):
                ps = np.round(evecs_sing[n_vec,bb,aa],6)
                if np.abs(ps) > 1e-6:
                    print(f'{ps}\t\t{mi_to_R[aa]}\t\t{np.round(np.sqrt(np.sum(bond[aa,:]**2)),3)}\t\t\t{bond[aa,:]}')

        vec = np.copy(irreps[n_vec,:,:])
        print('ir\tsh\tsp')
        while np.amax(np.abs(vec)) > 1e-5:
            (nn, aa) = np.unravel_index(
                    np.argmax(np.abs(vec)), irreps.shape[1:]
            )
            print(f'{nn}\t{map_shell[nn,aa]}\t{map_spat_irrep[nn,aa]}\t{np.round(vec[nn,aa],8)}')
            vec[nn,aa] = 0

        vec = np.round(irreps[n_vec,:,:],6)
        vec /= np.exp(1j*np.angle(vec.flatten()[np.argmax(np.abs(vec))]))

    return irreps