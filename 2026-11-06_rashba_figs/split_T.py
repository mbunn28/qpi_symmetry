# Rashba d+id at omega_2, full T at the resonance V0 = 1.33: split drho into the tau_0 and tau_z parts of
# T = t0 tau_0 + tz tau_z (drho is real-linear in T). Writes split_T.txt.
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
sys.path.insert(0, os.path.join(ROOT, '2026-11-05_fullT'))
from qpi_simulation import (PAULI, setup_mesh, neg_index, calculate_bloch, calculate_sc_order_params,  # noqa: E402
                            calculate_bdg, calculate_greens_function)
from qpi_fullT import fft_S, TAU_Z  # noqa: E402

RES = os.path.join(ROOT, '2026-11-05_fullT', 'data_nocut', 'delta0.1_gamma0.001_alpha0.1', 'dpid', 'res1',
                   'drho_om0.2095_V1.3279_q0inf.npz')
lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T
Nk, Nq = 240, 120  # as qpi_fullT.main


def overlap(x, y):
    return abs(np.vdot(x.ravel(), y.ravel())) / np.linalg.norm(x) / np.linalg.norm(y)


def main():
    k = setup_mesh(Nk, rlattice)
    en, u = np.linalg.eigh(calculate_bdg(calculate_bloch(-0.1, 0.1, k, lattice), calculate_bloch(-0.1, 0.1, -k, lattice),
                                         0.1*calculate_sc_order_params(k, lattice, 'd+id')))
    g0 = calculate_greens_function(np.array([0.98*np.amin(np.abs(en))]), 1e-3, en, u)[0]
    neg = neg_index(Nq)

    def drho(T):  # as qpi_fullT.tmatrix_qpi
        tr = np.einsum('mab,ijba->mij', PAULI, fft_S(g0, T, Nq, Nk//Nq)[..., :2, :2])
        return -(tr - np.conjugate(tr[:, neg][:, :, neg])) / (2j*np.pi)

    ref = np.load(RES)
    T = ref['T']
    t0, tz = np.trace(T)/4, np.trace(T @ TAU_Z)/4
    p0, pz, born = drho(t0*np.eye(4)), drho(tz*TAU_Z), drho(0.1*TAU_Z)
    with open(os.path.join(HERE, 'split_T.txt'), 'w') as f:
        f.write(f'V0 = {float(ref["v"]):.4f}, omega = {float(ref["omega"]):.4f}, t0 = {t0:.4g}, tz = {tz:.4g}, '
                f'max|T - t0 - tz tau_z| = {np.abs(T - t0*np.eye(4) - tz*TAU_Z).max():.1e}\n')
        f.write(f'max|p0 + pz - stored drho| / max = {np.abs(p0 + pz - ref["drho"]).max()/np.abs(ref["drho"]).max():.1e}\n')
        f.write('channel  max|tau0 part|  max|tauz part|  overlap(tau0, Born)  overlap(tauz, Born)\n')
        for m, ch in enumerate('0xyz'):
            f.write(f'{ch:>7}  {np.abs(p0[m]).max():14.3g}  {np.abs(pz[m]).max():14.3g}  '
                    f'{overlap(p0[m], born[m]):19.3f}  {overlap(pz[m], born[m]):19.3f}\n')



if __name__ == '__main__':
    main()
