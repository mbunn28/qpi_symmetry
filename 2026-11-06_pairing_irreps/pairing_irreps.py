# C6v irrep content of each pairing Delta(k) (2x2 in spin), with spin acting as with SOC:
# (g.Delta)(k) = D(g) Delta(R_g^-1 k) D(g)^T, D in SU(2) (its sign cancels). Writes pairing_irreps.txt.
import os
import sys
import numpy as np
import diverge.output as do  # type:ignore

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
import qpi_simulation as ans  # noqa: E402
import qpi_simulation_frg as frg  # noqa: E402

FRG_ROOT = '/Users/mtb31/Library/CloudStorage/OneDrive-Personal/Documents/StAndrews/triangular_rashba'
RUNS = [('nu0.5_U0_a0.1/V0.9_nk36_nkf25_c4.01', 'FRG U=0'), ('nu0.5_U8_a0.4/nk24_nkf31', 'FRG U=8, chi=+1')]
SIG = ans.PAULI
lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
rlattice = 2*np.pi * np.linalg.inv(lattice).T


# Spin conventions scanned: rotation sense (U[n] samples f(C6^n k), so g = C6^-n, or the reverse) and the
# spin part of the mirror S: y -> -y (C2 about y, its normal, or about x); D or D* (Delta vs Delta^* in the BdG).
CONVENTIONS = [(sense, axis, conj) for sense in (-1, 1) for axis in (2, 1) for conj in (False, True)]


def spin_ops(sense=-1, axis=2, conj=False):
    """D(g) in the element order of build_c6v_perms: C6^(sense n), then S C6^(sense n)."""
    rot = lambda th: np.cos(th/2)*SIG[0] - 1j*np.sin(th/2)*SIG[3]
    S = -1j*SIG[axis]
    ops = [rot(sense*n*np.pi/3) for n in range(6)] + [S @ rot(sense*n*np.pi/3) for n in range(6)]
    return [np.conj(D) for D in ops] if conj else ops


def weights(delta, kmesh, conv=(-1, 2, False)):
    """Weight of each irrep in Delta, shape (N, N, 2, 2); sums to 1."""
    U = frg.build_c6v_perms(kmesh)
    N = kmesh.shape[0]
    f = delta.reshape(N*N, 2, 2)
    chi = frg.c6v_character_table()
    proj = np.zeros((len(chi),) + f.shape, dtype=complex)
    for g, D in enumerate(spin_ops(*conv)):
        gf = D @ f[U[g]] @ D.T
        proj += np.multiply.outer(chi[:, g] * chi[:, 0] / len(U), gf)
    w = np.sum(np.abs(proj)**2, axis=(1, 2, 3))
    return w / np.sum(np.abs(f)**2)


def frg_delta(run, kmesh):
    run_dir = os.path.join(FRG_ROOT, run)
    name = os.path.basename(run_dir)
    O = do.read(os.path.join(run_dir, name + '.dvg'))
    M = do.read(os.path.join(run_dir, name + '.mod'))
    bond = frg.get_bond_vectors(O, np.array(M.lattice)[:2, :2])[:, 0, 0, :]
    ev = frg.calculate_sc_order_params(O, kmesh, bond)
    return ev[0] if len(ev) == 1 else (ev[0] + 1j*ev[1])/np.sqrt(2)


if __name__ == '__main__':
    kmesh = ans.setup_mesh(120, rlattice)
    cases = [(p, ans.calculate_sc_order_params(kmesh, lattice, p)[..., None, None] * 1j*SIG[2])
             for p in ['s', 'ext s', 'd+id', 'dx2y2']]
    cases += [(label, frg_delta(run, kmesh)) for run, label in RUNS]
    with open(os.path.join(HERE, 'pairing_irreps.txt'), 'w') as out:
        out.write('C6v irrep weights of Delta(k), spin acting as SU(2); singlet fraction = |psi|^2 / |Delta|^2\n')
        for label, d in cases:
            psi = np.einsum('...ab,ba->...', d, -1j*SIG[2]) / 2  # Delta = (psi + d.sigma) i sigma_y
            sing = 2*np.sum(np.abs(psi)**2) / np.sum(np.abs(d)**2)
            for conv in CONVENTIONS:
                w = weights(d, kmesh, conv)
                line = (f'{label:18s} {str(conv):18s} singlet {sing:.3f}  '
                        + '  '.join(f'{n} {x:.4f}' for n, x in zip(frg.IRREP_NAMES, w)) + f'  sum {w.sum():.6f}')
                print(line)
                out.write(line + '\n')
