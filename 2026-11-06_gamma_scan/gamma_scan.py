# d+id Rashba, full T at V0 = 0.1: max |drho_mu(q)| vs broadening Gamma, at omega = 0 and omega_2.
import os
import sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, R+'/2_new'); sys.path.insert(0, R+'/2026-11-05_fullT')
from qpi_simulation import PAULI, setup_mesh, neg_index, calculate_bloch, calculate_sc_order_params, calculate_bdg, calculate_greens_function
from qpi_fullT import fft_S, TAU_Z
lat = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]]); rl = 2*np.pi*np.linalg.inv(lat).T
Nk, Nq = 240, 120
k = setup_mesh(Nk, rl)
en, u = np.linalg.eigh(calculate_bdg(calculate_bloch(-0.1, 0.1, k, lat), calculate_bloch(-0.1, 0.1, -k, lat), 0.1*calculate_sc_order_params(k, lat, 'd+id')))
neg = neg_index(Nq)
for om in [0.0, 0.98*np.amin(np.abs(en))]:
    print(f'omega = {om:.4f}')
    for G in [2e-3, 1e-3, 5e-4, 2.5e-4]:
        g0 = calculate_greens_function(np.array([om]), G, en, u)[0]
        gl = g0.mean(axis=(0, 1)); V = 0.1*TAU_Z
        for name, T in [('born', V), ('full', np.linalg.solve(np.eye(4) - V @ gl, V))]:
            tr = np.einsum('mab,ijba->mij', PAULI, fft_S(g0, T, Nq, Nk//Nq)[..., :2, :2])
            d = np.abs(-(tr - np.conjugate(tr[:, neg][:, :, neg])) / (2j*np.pi)).reshape(4, -1).max(1)
            print(f'  Gamma={G:.2e} {name}: t0={np.trace(T)/4:.2e}  max drho 0,x,y,z = ' + ' '.join(f'{x:.3g}' for x in d) + f'   z/0 = {d[3]/d[0]:.3g}')
