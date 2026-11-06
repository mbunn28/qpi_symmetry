# Case table: for each (system, pairing, impurity), the irreps of each spin channel drho_mu(q), from saved Born data.
# An entry is the union over omega in {0, omega_2}; marked if it is finite at only one of them.
# The table is Born only; full-T entries that differ from Born go to case_table_fullT.txt.
import os
import re
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, '2_new'))
from qpi_simulation import setup_mesh, build_c6v_perms, irrep_decomposition, IRREP_NAMES  # noqa: E402

NOSOC = os.path.join(ROOT, '2026-11-05_nosoc', 'data', 'delta0.1_gamma0.001')
FULLT = os.path.join(ROOT, '2026-11-05_fullT', 'data')
FULLT_NOCUT = os.path.join(ROOT, '2026-11-05_fullT', 'data_nocut')  # full T: point impurity, no q cutoff
NODAL = os.path.join(HERE, 'data', 'A2_dx2y2_om0.1500_V0.1000_q020.0000.npz')
# pairing irreps from ../2026-11-06_pairing_irreps/pairing_irreps.txt
PAIR_TEX = {'s': '$A_1$: $s$', 'ext_s': r'$A_1$: ext.\ $s$', 'dpid': '$E_2$: $d+id$', 'dx2y2': '$E_2$: $d_{x^2-y^2}$'}
# FRG U, V from ../2026-11-06_pairing_irreps/frg_params.txt
FRG_TEX = [('frg_nu0.5_U0_a0.1_V0.9_nk36_nkf25_c4.01', r'$A_1$: $U{=}0, V{=}0.9$'),
           ('frg_nu0.5_U8_a0.4_nk24_nkf31', r'$E_2$: $U{=}8, V{=}0$, $\chi{=}+1$')]
SYSTEM_SUB = {'Rashba--Hubbard': '(FRG)'}  # second line of the System label
# breaking_TRS.tex figures plotting the QPI of each (system, impurity, pairing key); '(x)' = row label in that figure
FIGS = {**{('no SOC', 'A1', p): [f'nosoc_A1({r})'] for p, r in zip(PAIR_TEX, 'abcd')},
        ('no SOC', 'A2', 'dpid'): ['nosoc_A2'],
        ('Rashba', 'A1', 'dpid'): ['dpid_compare'],
        **{('Rashba', 'A1', p): [f'rashba_nonchiral({r})'] for p, r in zip(('s', 'ext_s', 'dx2y2'), 'abc')},
        ('Rashba--Hubbard', 'A1', 'frg_nu0.5_U8_a0.4_nk24_nkf31'): ['frg_spin'],
        ('Rashba--Hubbard', 'A1', 'frg_nu0.5_U0_a0.1_V0.9_nk36_nkf25_c4.01'): ['frg_nonchiral']}


def sorted_om(fs):
    return sorted(fs, key=lambda f: float(np.load(f)['omega']))


def cases():
    """(system, pairing tex, impurity, T, [omega=0 file, omega_2 file], pairing key); full T only for A1 (point impurity)."""
    out = []
    for imp, sub in [('A1', 'q0_1.00e+00'), ('A2', 'q0_2.00e+01')]:
        for p in PAIR_TEX:
            fs = sorted_om(glob.glob(os.path.join(NOSOC, imp, sub, p, 'drho_*.npz')))
            if p == 'dx2y2':  # nodal: omega_2 = 0.15, not the mesh "gap edge"
                fs = [fs[0], glob.glob(os.path.join(FULLT, 'delta0.1_gamma0.001_alpha0', 'dx2y2', 'born', 'drho_om0.1500*.npz'))[0]
                      if imp == 'A1' else NODAL]
            out.append(('no SOC', PAIR_TEX[p], imp, 'Born', fs, p))
            if imp == 'A1':
                out.append(('no SOC', PAIR_TEX[p], imp, 'full',
                            sorted_om(glob.glob(os.path.join(FULLT_NOCUT, 'delta0.1_gamma0.001_alpha0', p, 'full', 'drho_*.npz'))), p))
    for p in PAIR_TEX:
        for t, sub in [('Born', 'born'), ('full', 'full')]:
            out.append(('Rashba', PAIR_TEX[p], 'A1', t,
                        sorted_om(glob.glob(os.path.join(FULLT if sub == 'born' else FULLT_NOCUT, 'delta0.1_gamma0.001_alpha0.1', p, sub, 'drho_*.npz'))), p))
    for d, tex in FRG_TEX:
        for t, sub in [('Born', 'born'), ('full', 'full')]:
            out.append(('Rashba--Hubbard', tex, 'A1', t,
                        sorted_om(glob.glob(os.path.join(FULLT if sub == 'born' else FULLT_NOCUT, d, '*', '*', sub, 'drho_*.npz'))), d))
    return out


def irreps(f, mesh, U, min_weight=1e-3):
    """Per channel: set of (irrep, part) that are finite, part 'Re'/'Im'."""
    d = np.load(f)
    res = []
    for m in range(4):
        s = set()
        for part, img in [('Re', np.real(d['drho'][m])), ('Im', np.imag(d['drho'][m]))]:
            if np.amax(np.abs(img)) < d['zero_tol']:
                continue
            _, w, _ = irrep_decomposition(img, mesh, U)
            s |= {(IRREP_NAMES[n], part) for n in range(len(w)) if w[n] > min_weight}
        res.append(s)
    return res, float(d['omega'])


def entry(s0, s2):
    allr = s0 | s2
    if not allr:
        return '$0$'
    names = sorted({n for n, _ in allr}, key=IRREP_NAMES.index)
    parts = {p for _, p in allr}
    tex = ' + '.join(f'${n[0]}_{n[1]}$' for n in names)
    if parts == {'Im'}:
        tex = r'$\mathrm{Im}$: ' + tex
    if not s0:
        tex += r'$^\dagger$'
    elif not s2:
        tex += r'$^\ddagger$'
    return tex


if __name__ == '__main__':
    mesh = setup_mesh(120, 2*np.pi*np.linalg.inv(np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])).T)
    U = build_c6v_perms(mesh)
    rows = []
    for system, pairing, imp, t, fs, key in cases():
        assert len(fs) == 2, (system, pairing, imp, t)
        (r0, om0), (r2, om2) = irreps(fs[0], mesh, U), irreps(fs[1], mesh, U)
        assert om0 == 0
        cells = [entry(a, b) for a, b in zip(r0, r2)]
        figs = ', '.join(re.sub(r'^([^(]+)', r'\\ref{fig:\1}', x) for x in FIGS.get((system, imp, key), [])) or '--'
        rows.append((system, pairing, imp, t, f'{om2:.4f}', cells + [figs]))
        print(f'{system:16s} {pairing:34s} {imp} {t:4s} om2={om2:.4f}  ' + ' | '.join(cells))
    born = [r for r in rows if r[3] == 'Born']
    with open(os.path.join(HERE, 'case_table.tex'), 'w') as f:
        # Born only; columns: system, impurity, pairing, omega_2, 4 channels, figures; labels only on the first row of their block
        last, n_sys = (None, None), 0
        for system, pairing, imp, t, om2, cells in born:
            if last[0] is not None and system != last[0]:
                f.write('        \\hline\n')
            elif last[0] is not None and imp != last[1]:  # same system: rule spans columns 2-9, leaving System unbroken
                f.write('        \\cline{2-9}\n')
            n_sys = 0 if system != last[0] else n_sys + 1
            sys_cell = system if n_sys == 0 else (SYSTEM_SUB.get(system, '') if n_sys == 1 else '')
            f.write(f'        {sys_cell} & {imp if (system, imp) != last else ""} & {pairing} & {om2} & '
                    + ' & '.join(cells) + ' \\\\\n')
            last = (system, imp)
    # full T vs Born, for the caption and text (not in the table)
    ref = {r[:3]: r[5][:4] for r in born}
    with open(os.path.join(HERE, 'case_table_fullT.txt'), 'w') as f:
        f.write('Full T (v = 0.1) vs Born: entries of the case table that differ\n')
        for system, pairing, imp, t, om2, cells in rows:
            if t == 'full':
                diff = [f'{mu}: Born {a} -> full {b}' for mu, a, b in zip('0xyz', ref[(system, pairing, imp)], cells[:4]) if a != b]
                f.write(f'{system:16s} {pairing:34s} {imp}  ' + ('; '.join(diff) if diff else 'same') + '\n')
