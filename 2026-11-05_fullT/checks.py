# Validation of the full-T pipeline; writes checks.txt. Run after qpi_fullT.py and qpi_fullT_frg.py.
#  1. fft_S equals 2_new born_S (small mesh, any T).
#  2. Born outputs equal the existing 2_new runs (ansatz and FRG) and the 2026-11-05_nosoc run (dx2y2, omega = 0).
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
from qpi_simulation import (setup_mesh, calculate_bloch, calculate_sc_order_params, calculate_bdg,  # noqa: E402
                            calculate_greens_function, born_S)
from qpi_fullT import fft_S, TAU_Z  # noqa: E402


def rel(a, b):
    return np.amax(np.abs(a - b)) / np.amax(np.abs(b))


def check_fft_S(out, Nk=48, Nq=24):
    latt = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    b = 2*np.pi*np.linalg.inv(latt).T
    k = setup_mesh(Nk, b)
    for p in ['s', 'd+id']:
        en, u = np.linalg.eigh(calculate_bdg(calculate_bloch(-0.1, 0.1, k, latt), calculate_bloch(-0.1, 0.1, -k, latt),
                                             0.1*calculate_sc_order_params(k, latt, p)))
        g0 = calculate_greens_function(np.array([0.05]), 1e-3, en, u)[0]
        T = np.linalg.solve(np.eye(4) - 0.1*TAU_Z @ g0.mean(axis=(0, 1)), 0.1*TAU_Z)
        out.write(f'fft_S vs born_S, {p}, Nk={Nk}, Nq={Nq}, full T: rel diff {rel(fft_S(g0, T, Nq, Nk//Nq), born_S(g0, T, Nq, Nk//Nq)):.1e}\n')


def check_born(out):
    pairs = []
    a = os.path.join(HERE, 'data', 'delta0.1_gamma0.001_alpha0.1')
    for p in ['s', 'ext_s', 'dpid']:  # dx2y2: 2_new used the spurious mesh gap, only omega = 0 comparable
        for f in glob.glob(os.path.join(a, p, 'born', 'drho_*.npz')):
            pairs.append((f, os.path.join(ROOT, '2_new', 'data', 'delta0.1_gamma0.001_alpha0.1', 'q0_1.00e+00', p, os.path.basename(f))))
    pairs.append((os.path.join(a, 'dx2y2', 'born', 'drho_om0.0000_V0.1000_q01.0000.npz'),
                  os.path.join(ROOT, '2_new', 'data', 'delta0.1_gamma0.001_alpha0.1', 'q0_1.00e+00', 'dx2y2', 'drho_om0.0000_V0.1000_q01.0000.npz')))
    for f in glob.glob(os.path.join(HERE, 'data', 'frg_*', '*', '*', 'born', 'drho_*.npz')):
        parts = f.split(os.sep)
        pairs.append((f, os.path.join(ROOT, '2_new', 'data', parts[-5], parts[-4], 'q0_1.00e+00', parts[-3], parts[-1])))
    pairs.append((os.path.join(HERE, 'data', 'delta0.1_gamma0.001_alpha0', 'dx2y2', 'born', 'drho_om0.0000_V0.1000_q01.0000.npz'),
                  os.path.join(ROOT, '2026-11-05_nosoc', 'data', 'delta0.1_gamma0.001', 'A1', 'q0_1.00e+00', 'dx2y2', 'drho_om0.0000_V0.1000_q01.0000.npz')))
    for f, g in sorted(pairs):
        out.write(f'Born {os.path.relpath(f, HERE)} vs {os.path.relpath(g, ROOT)}: rel diff {rel(np.load(f)["drho"], np.load(g)["drho"]):.1e}\n')


if __name__ == '__main__':
    with open(os.path.join(HERE, 'checks.txt'), 'w') as out:
        check_fft_S(out)
        check_born(out)
    print(open(os.path.join(HERE, 'checks.txt')).read())
