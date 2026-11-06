# Full (local) T-matrix for the A1 impurity, method of soc_materials/spin_qpi/qpi_fritz.py:
#   T(w) = [1 - V G_loc(w)]^-1 V, G_loc = sum_k g0(k)/N^2, V = v tau_z,
# with resonant v = 1/Re(lambda), lambda eigenvalues of G_loc(w, Gamma=0) tau_z.
# Born (T = V) is run alongside as the reference. Rest of the pipeline as 2_new/qpi_simulation.py.
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
from qpi_simulation import (PAULI, CHANNELS, setup_mesh, neg_index, fold_to_ws, build_c6v_perms,  # noqa: E402
                            calculate_bloch, calculate_sc_order_params, calculate_bdg, calculate_greens_function,
                            irrep_decomposition, write_irrep_analysis, plot_drho, plot_irreps, plot_note,
                            high_symmetry_path, plot_bands)

TAU_Z = np.diag([1, 1, -1, -1])


def fft_S(g0, T, Nq, qstep):
    """
    S(q) = sum_k g0(k) T g0(k-q), same index convention as born_S, via the convolution theorem
    (the real-space product G(r) T G(-r) of qpi_fritz.py). Any q-independent T.
    """
    N = g0.shape[0]
    A = np.fft.fft2(g0, axes=(0, 1))
    B = np.fft.ifft2(g0, axes=(0, 1)) * N**2
    C = np.fft.ifft2(A @ T @ B, axes=(0, 1))
    s = ((np.arange(Nq) - Nq//2) * qstep) % N
    return C[np.ix_(s, s)]


def resonant_v(g_loc0, tol=1e-6):
    """
    Distinct 1/lambda for real lambda in eig(G_loc(Gamma=0) tau_z). G_loc tau_z is not Hermitian when
    the local anomalous part is non-zero (s, ext s), so eig, not eigh; complex lambda (no bound state) are returned separately.
    """
    lam = np.linalg.eigvals(g_loc0 @ TAU_Z)
    real = np.abs(np.imag(lam)) < tol*np.abs(lam)
    v = []
    for x in np.sort(1/np.real(lam[real])):
        if not v or abs(x - v[-1]) > tol*abs(x):
            v.append(x)
    return np.array(v), lam[~real]


def tmatrix_qpi(en, u, gap_size, base_dir, pairing_name, header, title_head, gamma, v_mag,
                Nk, Nq, q0_factor=1.0, r_window=10, check_born=False, omega_vals=None, resonances=True):
    """
    Born, full T at v_mag and, if resonances (only meaningful for a true gap), full T at each resonant v, at omega_vals (default {0, 0.98 gap}), from BdG eigensystem
    (en, u) on the standard Nk mesh. header: parameter line for the reports; title_head: start of plot titles.
    """
    if Nk % Nq != 0:
        raise ValueError('Nk must be a multiple of Nq so k-q lies on the k-mesh')
    qstep = Nk // Nq
    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq, Ur = build_c6v_perms(qmesh), build_c6v_perms(rmesh)
    neg = neg_index(Nq)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))
    # q0_factor None: no cutoff, the consistent choice for a point impurity (T is q-independent)
    if q0_factor is None:
        cutoff, q0_tag, q0_tex = 1.0, 'inf', r'\infty'
    else:
        q0 = np.linalg.norm(rlattice[0]) * q0_factor
        cutoff = np.exp(-np.sum(fold_to_ws(qmesh, rlattice)**2, axis=-1) / q0**2)[:, :, None, None]
        q0_tag = q0_tex = f'{q0_factor:.4f}'

    omega_vals = np.array([0, 0.98*gap_size]) if omega_vals is None else np.asarray(omega_vals)
    g0k = calculate_greens_function(omega_vals, gamma, en, u)
    g_loc = np.mean(g0k, axis=(1, 2))
    # as qpi_fritz.py: resonance from the clean (Gamma = 0) G_loc, T from the broadened one
    g_loc0 = np.mean(calculate_greens_function(omega_vals, 0, en, u), axis=(1, 2))

    summary = open(os.path.join(base_dir, f'tmatrix_{pairing_name}.txt'), 'w')
    summary.write(f'{header}\nNk={Nk}\n')

    for ii, om in enumerate(omega_vals):
        v_res, lam_complex = resonant_v(g_loc0[ii]) if resonances else (np.array([]), 'not considered (no gap)')
        summary.write(f'\nomega={om:.6f}\n  G_loc =\n{np.array2string(g_loc[ii], precision=4)}\n')
        summary.write(f'  resonant v = {v_res}; complex lambda (skipped) = {lam_complex}\n')

        cases = [('born', v_mag, v_mag*TAU_Z), ('full', v_mag, None)] + \
                [(f'res{n}', v, None) for n, v in enumerate(v_res)]
        for kind, v, T in cases:
            if T is None:
                V = v*TAU_Z
                T = np.linalg.solve(np.eye(4) - V @ g_loc[ii], V)
            summary.write(f'  {kind}: v={v:.6f}, max|T|={np.amax(np.abs(T)):.4e}, '
                          f'min|eig(1-VG)|={np.amin(np.abs(np.linalg.eigvals(np.eye(4) - v*TAU_Z @ g_loc[ii]))):.3e}\n')

            S = fft_S(g0k[ii], T, Nq, qstep)
            if check_born:
                from qpi_simulation import born_S
                ref = born_S(g0k[ii], T, Nq, qstep)
                print(f'  {kind}: max|fft_S - born_S| / max|born_S| = {np.amax(np.abs(S - ref))/np.amax(np.abs(ref)):.2e}')
            S = S * cutoff
            zero_tol = 1e-10 * np.amax(np.abs(T)) * np.sum(np.abs(g0k[ii])**2)

            out_dir = os.path.join(base_dir, pairing_name, kind)
            os.makedirs(out_dir, exist_ok=True)
            tag = f'om{om:.4f}_V{v:.4f}_q0{q0_tag}'
            t_tex = 'Born' if kind == 'born' else 'full $T$'
            title = (
                rf'{title_head}, {t_tex}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}'
                '\n' + rf'$\omega=${om:.4f}, $V=${v:.4f}, $q_0={q0_tex} \times |b_1|$'
            )

            # drho_mu(q) = -[Tr sigma_mu S(q) - Tr sigma_mu S(-q)^+]/(2 pi i), particle block only
            tr = np.einsum('mab,ijba->mij', PAULI, S[..., :2, :2])
            drho = -(tr - np.conjugate(tr[:, neg][:, :, neg])) / (2j*np.pi)
            drho_r = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(drho, axes=(1, 2))), axes=(1, 2))

            with open(os.path.join(out_dir, f'symmetry_analysis_{tag}.txt'), 'w') as report:
                report.write(f'Symmetry analysis, A1 impurity, {kind} T-matrix\n{header}\n')
                report.write(f'omega={om}, v_mag={v}, q0_factor={q0_factor}, Nk={Nk}, Nq={Nq}\n')
                report.write(f'T =\n{np.array2string(T, precision=4)}\n\n')
                report.write('C6v irrep projection (weight = sum|f_nu|^2 / sum|f|^2):\n')
                report.write(f'Datasets with max |f| < {zero_tol:.2e} are numerically zero\n')

                datasets = []
                for m, ch in enumerate(CHANNELS):
                    sub = rf'_{ch}' if ch != '0' else ''
                    for name, label, img, dkind, U in [
                        (f'Re drho_{ch}(q)', rf'$\mathrm{{Re}}\,\delta\rho{sub}(\bvec{{q}})$', np.real(drho[m]), 'q', Uq),
                        (f'Im drho_{ch}(q)', rf'$\mathrm{{Im}}\,\delta\rho{sub}(\bvec{{q}})$', np.imag(drho[m]), 'q', Uq),
                        (f'drho_{ch}(r)', rf'$\delta\rho{sub}(\bvec{{r}})$', np.real(drho_r[m]), 'r', Ur),
                    ]:
                        fnu, weights, maxvals = irrep_decomposition(img, geom[dkind][0], U)
                        negligible = np.amax(np.abs(img)) < zero_tol
                        if negligible:
                            weights = np.zeros_like(weights)
                        datasets.append(dict(name=name, label=label, img=img, kind=dkind,
                                             fnu=fnu, weights=weights, maxvals=maxvals, negligible=negligible))
                    report.write(f'\nChannel sigma_{ch}:\n')
                    for d in datasets[-3:]:
                        write_irrep_analysis(report, d['name'], d['weights'], d['maxvals'], d['negligible'])
                    report.write(f'Check: max |Im drho_{ch}(r)| = {np.amax(np.abs(np.imag(drho_r[m]))):.4e}\n')

            plot_drho(datasets, geom, r_window, os.path.join(out_dir, f'drho_{tag}.png'), title)
            plot_irreps(datasets, geom, r_window, os.path.join(out_dir, f'irreps_{tag}.png'), title)
            plot_note(datasets, geom, r_window, os.path.join(out_dir, f'note_drho_{tag}.pdf'))
            np.savez(os.path.join(out_dir, f'drho_{tag}.npz'), drho=drho, drho_r=drho_r, zero_tol=zero_tol,
                     omega=om, gap_size=gap_size, T=T, v=v)
    summary.close()


def main(pairing, mu=-0.1, alpha=0.1, delta_mag=1e-1, gamma=1e-3, v_mag=1e-1,
         Nk=240, Nq=120, q0_factor=1.0, r_window=10, data_root=None, check_born=False, omega_vals=None, resonances=True):
    """Rashba ansatz pairing; omega_vals overrides {0, 0.98 gap}. Nodal states: set omega_vals, resonances=False (mesh gap is spurious). q0 = q0_factor * |b1|. check_born compares fft_S with born_S (slow; smoke tests only)."""
    data_root = data_root or os.path.join(HERE, 'data')
    base_dir = os.path.join(data_root, f'delta{delta_mag:g}_gamma{gamma:g}_alpha{alpha:g}')
    pairing_name = pairing.replace('+', 'p').replace(' ', '_')
    os.makedirs(base_dir, exist_ok=True)

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(Nk, rlattice)

    h = calculate_bloch(mu, alpha, kmesh, lattice)
    op = delta_mag * calculate_sc_order_params(kmesh, lattice, pairing)
    en, u = np.linalg.eigh(calculate_bdg(h, calculate_bloch(mu, alpha, -kmesh, lattice), op))
    gap_size = np.amin(np.abs(en))
    print(f'{pairing}: gap size {gap_size}')

    bands_path = os.path.join(base_dir, f'bands_{pairing_name}.png')
    if not os.path.exists(bands_path):
        kpath, dist, ticks, labels = high_symmetry_path(rlattice)
        h_path = calculate_bloch(mu, alpha, kpath[None], lattice)
        en_path = np.linalg.eigvalsh(calculate_bdg(
            h_path, calculate_bloch(mu, alpha, -kpath[None], lattice),
            delta_mag * calculate_sc_order_params(kpath[None], lattice, pairing)))
        plot_bands(dist, ticks, labels, np.linalg.eigvalsh(h_path)[0], en_path[0], bands_path,
                   rf'{pairing} pairing, $\alpha$ = {alpha:g}, $\mu$ = {mu:g}, $|\Delta|$ = {delta_mag:g}, gap = {gap_size:.4f}')
    if gap_size < 1e-6:
        with open(os.path.join(base_dir, f'no_gap_{pairing_name}.txt'), 'w') as f:
            f.write(f'No gap found for mu {mu}, alpha {alpha}, gamma {gamma}, delta_mag {delta_mag}\n')
        return

    header = f'{pairing} pairing: mu={mu}, alpha={alpha}, delta_mag={delta_mag}, gamma={gamma}, gap_size={gap_size}'
    title_head = rf'{pairing} pairing, $\alpha$ = {alpha:g}, $|\Delta|$ = {delta_mag:.2e}'
    tmatrix_qpi(en, u, gap_size, base_dir, pairing_name, header, title_head, gamma, v_mag,
                Nk, Nq, q0_factor, r_window, check_born, omega_vals, resonances)


def nosoc_gapped(**kw):
    """No-SOC s, ext s, d+id at {0, 0.98 gap}, for the case table only (no resonance search)."""
    for p in ['s', 'ext s', 'd+id']:
        main(p, alpha=0.0, resonances=False, **kw)


def run_all(**kw):
    for p in ['s', 'ext s', 'd+id']:
        main(p, **kw)
    # dx2y2 is nodal (2026-11-06_dx2y2_nodes): fixed omega_2 = 0.15 ~ half the max FS gap, no resonances
    main('dx2y2', omega_vals=[0, 0.15], resonances=False, **kw)
    main('dx2y2', alpha=0.0, omega_vals=[0, 0.15], resonances=False, **kw)
    nosoc_gapped(**kw)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        for p in ['s', 'd+id']:
            main(p, Nk=48, Nq=24, data_root=sys.argv[2], check_born=True)
    elif len(sys.argv) > 1 and sys.argv[1] == 'nosoc':
        nosoc_gapped()
    elif len(sys.argv) > 1 and sys.argv[1] == 'nocut':
        run_all(q0_factor=None, data_root=os.path.join(HERE, 'data_nocut'))
    else:
        run_all()
