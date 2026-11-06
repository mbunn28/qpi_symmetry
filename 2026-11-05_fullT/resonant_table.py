# Rows of the resonant-case table (tab:resonant of breaking_TRS.tex) from the res* outputs; writes resonant_table.tex.
# Ratios are of q-space maxima of |drho_mu(q)| (q = 0 included). Asserts the caption's irreps instead of tabulating them:
# Re drho_0, Re drho_z in A1, Im drho_{x,y} in E1 (or zero), the other parts zero.
import os
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = [('delta0.1_gamma0.001_alpha0.1', 'dpid', r'$d+id$ (Rashba)'),
          ('delta0.1_gamma0.001_alpha0.1', 'ext_s', r'ext.\ $s$ (Rashba)'),
          ('frg_nu0.5_U8_a0.4_nk24_nkf31/delta_rms0.1_gamma0.001', 'frg_chi+1', r'FRG $U = 8$, $\chi = +1$')]


EXPECTED = {'Re drho_0(q)': 'A1', 'Im drho_0(q)': None, 'Re drho_x(q)': None, 'Im drho_x(q)': 'E1',
            'Re drho_y(q)': None, 'Im drho_y(q)': 'E1', 'Re drho_z(q)': 'A1', 'Im drho_z(q)': None}


def check_irreps(f, min_weight=1e-3):
    """Each q-dataset is zero or (if an irrep is expected) purely that irrep, per the symmetry_analysis report next to f."""
    rep = open(f.replace('drho_', 'symmetry_analysis_').replace('.npz', '.txt')).read()
    for name, irrep in EXPECTED.items():
        block = rep.split(name + ':')[1].split('\n')
        if 'numerically zero' in block[0]:
            continue
        w = [l.split()[0].rstrip(':') for l in block[1:7] if float(l.split('=')[1].split(',')[0]) > min_weight]
        assert w == [irrep], f'{f}: {name} has {w}, expected {irrep}'


def fmt(x):
    if x < 1e-8:
        return '$0$'
    e = int(np.floor(np.log10(x)))
    return rf'${x/10**e:.1f}\times10^{{{e}}}$'


if __name__ == '__main__':
    lines = []  # omega = 0 printed as 0.0000
    for base, p, tex in STATES:
        rows = []
        for f in glob.glob(os.path.join(HERE, 'data_nocut', base, p, 'res*', 'drho_*.npz')):
            d = np.load(f)
            m = np.amax(np.abs(d['drho']), axis=(1, 2))
            om, v = float(d['omega']), float(d['v'])
            check_irreps(f)
            rows.append((om, v, np.amax(np.abs(d['T'])), m[3]/m[0], max(m[1], m[2])/m[0]))
        for i, (om, v, t, z, xy) in enumerate(sorted(rows)):
            lines.append(f'        {tex if i == 0 else "":24s} & ${om:.4f}$ & ${v:.2f}$ & ${t:.3g}$ & {fmt(z)} & {fmt(xy)} \\\\')
    open(os.path.join(HERE, 'resonant_table.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
