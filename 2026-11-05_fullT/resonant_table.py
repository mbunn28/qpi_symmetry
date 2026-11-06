# Rows of the resonant-case table (Table I of breaking_TRS.tex) from the res* outputs; writes resonant_table.tex.
# Ratios are of q-space maxima of |drho_mu(q)|; min|eig| = min |eig(1 - V G_loc)| at Gamma = 1e-3.
import os
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = [('delta0.1_gamma0.001_alpha0.1', 'dpid', r'$d+id$ (Rashba)'),
          ('delta0.1_gamma0.001_alpha0.1', 'ext_s', r'ext.\ $s$ (Rashba)'),
          ('frg_nu0.5_U8_a0.4_nk24_nkf31/delta_rms0.1_gamma0.001', 'frg_chi+1', r'FRG $U = 8$, $\chi = +1$')]


def min_eig(summary, om, v):
    """min|eig(1-VG)| for this omega and v, from tmatrix_<pairing>.txt."""
    block = summary.split(f'omega={om:.6f}')[1].split('omega=')[0]
    for line in block.splitlines():
        if f'v={v:.6f}' in line and line.strip().startswith('res'):
            return float(line.split('min|eig(1-VG)|=')[1])
    raise ValueError(f'no resonance v={v} at omega={om}')


def drho0_irreps(f, min_weight=1e-3):
    """Irreps of Re drho_0(q) with weight > min_weight, from the symmetry_analysis report next to f."""
    rep = open(f.replace('drho_', 'symmetry_analysis_').replace('.npz', '.txt')).read()
    block = rep.split('Re drho_0(q):')[1].split('Im drho_0(q)')[0]
    w = [l.split()[0].rstrip(':') for l in block.splitlines()[1:] if 'weight' in l and float(l.split('=')[1].split(',')[0]) > min_weight]
    return ' + '.join(f'${n[0]}_{n[1]}$' for n in w)


def fmt(x):
    if x < 1e-8:
        return '$0$'
    e = int(np.floor(np.log10(x)))
    return rf'${x/10**e:.1f}\times10^{{{e}}}$'


if __name__ == '__main__':
    lines = []  # omega = 0 printed as 0.0000
    for base, p, tex in STATES:
        summary = open(glob.glob(os.path.join(HERE, 'data_nocut', base, f'tmatrix_{p}.txt'))[0]).read()
        rows = []
        for f in glob.glob(os.path.join(HERE, 'data_nocut', base, p, 'res*', 'drho_*.npz')):
            d = np.load(f)
            m = np.amax(np.abs(d['drho']), axis=(1, 2))
            om, v = float(d['omega']), float(d['v'])
            rows.append((om, v, np.amax(np.abs(d['T'])), min_eig(summary, om, v), drho0_irreps(f), m[3]/m[0], max(m[1], m[2])/m[0]))
        for i, (om, v, t, e, ir, z, xy) in enumerate(sorted(rows)):
            lines.append(f'        {tex if i == 0 else "":24s} & ${om:.4f}$ & ${v:.2f}$ & ${t:.3g}$ & ${e:.2g}$ & {ir} & {fmt(z)} & {fmt(xy)} \\\\')
    open(os.path.join(HERE, 'resonant_table.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
