# Parameters of the two FRG runs used in the note, read from their .input files, and the filling nu
# (fraction of normal-state states below mu, both Rashba bands) from the hr.dat and mu the QPI code uses. Writes frg_params.txt.
import os
import sys
import numpy as np
import diverge.output as do  # type:ignore

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
import qpi_simulation_frg as frg  # noqa: E402
from pairing_irreps import FRG_ROOT, RUNS  # noqa: E402

if __name__ == '__main__':
    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    kmesh = frg.setup_mesh(600, 2*np.pi*np.linalg.inv(lattice).T)
    with open(os.path.join(HERE, 'frg_params.txt'), 'w') as out:
        for run, label in RUNS:
            run_dir = os.path.join(FRG_ROOT, run)
            name = os.path.basename(run_dir)
            inp = os.path.join(run_dir, name + '.input')
            keys = {}
            for k in ['tbfile', 'U', 'V', 'J', 'chemical_potential']:
                try:
                    keys[k] = frg.read_input_key(inp, k)
                except KeyError:
                    keys[k] = 'absent'
            M = do.read(os.path.join(run_dir, name + '.mod'))
            ff_wan, t, _ = frg.bloch_ff_from_w90(os.path.join(run_dir, keys['tbfile']))
            mu = float(keys['chemical_potential'])
            e = np.linalg.eigvalsh(frg.calculate_bloch(ff_wan, t, mu, kmesh, np.array(M.lattice)[:2, :2]))
            line = f'{label:16s} {run:38s} ' + '  '.join(f'{k}={v}' for k, v in keys.items()) + f'  nu={np.mean(e < 0):.4f}'
            print(line)
            out.write(line + '\n')
