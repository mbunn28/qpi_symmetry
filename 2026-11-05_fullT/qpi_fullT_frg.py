# Full T-matrix QPI (qpi_fullT.tmatrix_qpi) for the divERGe FRG gaps; BdG built as in 2_new/qpi_simulation_frg.py.
import os
import sys
import numpy as np
import diverge.output as do #type:ignore

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
import qpi_simulation_frg as frg  # noqa: E402
from qpi_fullT import tmatrix_qpi  # noqa: E402

FRG_ROOT = '/Users/mtb31/Library/CloudStorage/OneDrive-Personal/Documents/StAndrews/triangular_rashba'
RUNS = [os.path.join(FRG_ROOT, 'nu0.5_U8_a0.4/nk24_nkf31'),
        os.path.join(FRG_ROOT, 'nu0.5_U0_a0.1/V0.9_nk36_nkf25_c4.01')]


def main(run_dir, chirality=1, delta_mag=1e-1, gamma=1e-3, v_mag=1e-1, Nk=240, Nq=120, q0_factor=1.0,
         data_root=None, check_born=False):
    """delta_mag = BZ rms of the singlet-equivalent gap; Delta = (v0 + i chirality v1)/sqrt2 for a degenerate pair."""
    name = os.path.basename(os.path.normpath(run_dir))
    run_tag = f'{os.path.basename(os.path.dirname(os.path.normpath(run_dir)))}_{name}'
    base_dir = os.path.join(data_root or os.path.join(HERE, 'data'), f'frg_{run_tag}', f'delta_rms{delta_mag:g}_gamma{gamma:g}')
    os.makedirs(base_dir, exist_ok=True)

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = frg.setup_mesh(Nk, rlattice)

    O = do.read(os.path.join(run_dir, name + '.dvg'))
    M = do.read(os.path.join(run_dir, name + '.mod'))
    if not np.allclose(M.kmesh[0], 0) or O.n_orb != 1 or O.n_spin != 2:
        raise ValueError('Expected q-index 0 at Gamma, one orbital and n_spin = 2')
    latt_frg = np.array(M.lattice)[:2, :2]
    bond = frg.get_bond_vectors(O, latt_frg)[:, 0, 0, :]
    input_file = os.path.join(run_dir, name + '.input')
    ff_wan, t, _ = frg.bloch_ff_from_w90(os.path.join(run_dir, frg.read_input_key(input_file, 'tbfile')))
    mu = float(frg.read_input_key(input_file, 'chemical_potential'))

    evecs_k = frg.calculate_sc_order_params(O, kmesh, bond)
    chiral = len(evecs_k) == 2
    delt = evecs_k[0] if not chiral else (evecs_k[0] + 1j*chirality*evecs_k[1])/np.sqrt(2)
    h = frg.calculate_bloch(ff_wan, t, mu, kmesh, latt_frg)
    en, u = np.linalg.eigh(frg.calculate_bdg(h, frg.calculate_bloch(ff_wan, t, mu, -kmesh, latt_frg), delta_mag*delt))
    gap_size = np.amin(np.abs(en))
    pairing_name = 'frg' + (f'_chi{chirality:+d}' if chiral else '')
    print(f'{run_tag} ({pairing_name}): gap size {gap_size}')
    if gap_size < 1e-6:
        with open(os.path.join(base_dir, f'no_gap_{pairing_name}.txt'), 'w') as f:
            f.write(f'No gap found for {run_dir}, chirality {chirality}, gamma {gamma}, delta_mag {delta_mag}\n')
        return

    header = (f'FRG {run_tag}, chirality={chirality if chiral else "n/a"}: mu={mu} (divERGe), '
              f'delta_rms={delta_mag}, gamma={gamma}, gap_size={gap_size}; source {run_dir}')
    title_head = 'FRG ' + run_tag.replace('_', r'\_') + (rf', $\chi$ = {chirality:+d}' if chiral else '') + \
                 rf', $\Delta_{{\rm rms}}$ = {delta_mag:.2e}'
    tmatrix_qpi(en, u, gap_size, base_dir, pairing_name, header, title_head, gamma, v_mag,
                Nk, Nq, q0_factor, check_born=check_born)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        for r in RUNS:
            main(r, Nk=48, Nq=24, data_root=sys.argv[2], check_born=True)
    elif len(sys.argv) > 1 and sys.argv[1] == 'nocut':
        for r in RUNS:
            main(r, q0_factor=None, data_root=os.path.join(HERE, 'data_nocut'))
    else:
        for r in RUNS:
            main(r)
