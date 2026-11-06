# No-SOC rerun of 1_basic with the 2_new machinery: spinful BdG at alpha = 0, spin-resolved drho_mu.
# Impurities as in 1_basic: 'A1' scalar, T = v tau_z; 'A2', T = i v (k x k')_z tau_z.
import os
import sys
import numpy as np
from tqdm import tqdm #type:ignore

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '2_new'))
from qpi_simulation import (PAULI, CHANNELS, setup_mesh, neg_index, fold_to_ws, build_c6v_perms,  # noqa: E402
                            calculate_bloch, calculate_sc_order_params, calculate_bdg, calculate_greens_function,
                            born_S, irrep_decomposition, write_irrep_analysis, plot_drho, plot_irreps,
                            plot_note,
                            high_symmetry_path, plot_bands)

TAU_Z = np.diag([1, 1, -1, -1])



def a2_S(g0, kws, v_mag, Nq, qstep):
    """S(q) = sum_k g0(k) V(k, k-q) tau_z g0(k-q), V = i v (k x (k-q))_z with both momenta folded to the WS cell."""
    gM = g0 @ TAU_Z
    nb = g0.shape[-1]
    S = np.zeros((Nq, Nq, nb, nb), dtype=complex)
    for i, j in tqdm(np.ndindex(Nq, Nq), total=Nq*Nq, desc='QPI loop (A2)'):
        shift = ((i - Nq//2)*qstep, (j - Nq//2)*qstep)
        kmq = np.roll(kws, shift, axis=(0, 1))
        V = v_mag * 1j * (kws[..., 0]*kmq[..., 1] - kws[..., 1]*kmq[..., 0])
        S[i, j] = np.tensordot(V[..., None, None] * gM, np.roll(g0, shift, axis=(0, 1)), axes=([0, 1, 3], [0, 1, 2]))
    return S


def main(pairing, impurity='A1', mu=-0.1, delta_mag=1.0, gamma=1e-2, v_mag=1e-1,
         Nk=120, Nq=120, q0_factors=(1.0,), r_window=10):
    """q0 = q0_factor * |b1|; r_window is the plotted real-space radius in lattice constants."""
    alpha = 0.0
    qstep = Nk // Nq
    imp_tag = impurity
    sav_dir = os.path.join(HERE, 'data', f'delta{delta_mag:g}_gamma{gamma:g}', imp_tag)
    pairing_name = pairing.replace('+', 'p').replace(' ', '_')
    os.makedirs(sav_dir, exist_ok=True)

    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(Nk, rlattice)
    qmesh = setup_mesh(Nq, rlattice)
    rmesh = setup_mesh(Nq, lattice) * Nq
    Uq, Ur = build_c6v_perms(qmesh), build_c6v_perms(rmesh)
    neg = neg_index(Nq)
    geom = dict(q=(qmesh, rlattice), r=(rmesh, lattice))
    q_sq = np.sum(fold_to_ws(qmesh, rlattice)**2, axis=-1)
    kws = fold_to_ws(kmesh, rlattice)

    h = calculate_bloch(mu, alpha, kmesh, lattice)
    op = delta_mag * calculate_sc_order_params(kmesh, lattice, pairing)
    en, u = np.linalg.eigh(calculate_bdg(h, calculate_bloch(mu, alpha, -kmesh, lattice), op))
    gap_size = np.amin(np.abs(en))
    print(f'{pairing}, {imp_tag}: gap size {gap_size}')

    bands_path = os.path.join(HERE, 'data', f'delta{delta_mag:g}_gamma{gamma:g}', f'bands_{pairing_name}.png')
    if not os.path.exists(bands_path):
        kpath, dist, ticks, labels = high_symmetry_path(rlattice)
        h_path = calculate_bloch(mu, alpha, kpath[None], lattice)
        en_path = np.linalg.eigvalsh(calculate_bdg(
            h_path, calculate_bloch(mu, alpha, -kpath[None], lattice),
            delta_mag * calculate_sc_order_params(kpath[None], lattice, pairing)))
        plot_bands(dist, ticks, labels, np.linalg.eigvalsh(h_path)[0], en_path[0], bands_path,
                   rf'{pairing} pairing, no SOC, $\mu$ = {mu:g}, $|\Delta|$ = {delta_mag:g}, gap = {gap_size:.4f}')

    omega_vals = np.array([0, 0.98*gap_size])
    g0k = calculate_greens_function(omega_vals, gamma, en, u)

    for ii, om in enumerate(omega_vals):
        if impurity == 'A1':
            S_inf = born_S(g0k[ii], v_mag * TAU_Z, Nq, qstep)
            bound = v_mag
        else:
            S_inf = a2_S(g0k[ii], kws, v_mag, Nq, qstep)
            bound = v_mag * np.amax(np.sum(kws**2, axis=-1))
        zero_tol = 1e-10 * bound * np.sum(np.abs(g0k[ii])**2)

        for q0_factor in q0_factors:
            out_dir = os.path.join(sav_dir, f'q0_{q0_factor:.2e}', pairing_name)
            os.makedirs(out_dir, exist_ok=True)
            tag = f'om{om:.4f}_V{v_mag:.4f}_q0{q0_factor:.4f}'
            imp_tex = 'scalar ($A_1$) impurity' if impurity == 'A1' else r'$A_2$ impurity'
            title = (
                rf'{pairing} pairing, no SOC, {imp_tex}, $|\Delta|$ = {delta_mag:.2e}, $\Gamma=$ {gamma:.2e}, Gap = {gap_size:.4f}'
                '\n' + rf'$\omega=${om:.4f}, $V=${v_mag:.4f}, $q_0={q0_factor:.4f} \times |b_1|$'
            )

            q0 = np.linalg.norm(rlattice[0])*q0_factor
            S = S_inf * np.exp(-q_sq / q0**2)[:, :, None, None]

            # drho_mu(q) = -[Tr sigma_mu S(q) - Tr sigma_mu S(-q)^+]/(2 pi i), particle block only
            tr = np.einsum('mab,ijba->mij', PAULI, S[..., :2, :2])
            drho = -(tr - np.conjugate(tr[:, neg][:, :, neg])) / (2j*np.pi)
            drho_r = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(drho, axes=(1, 2))), axes=(1, 2))

            with open(os.path.join(out_dir, f'symmetry_analysis_{tag}.txt'), 'w') as report:
                report.write(f'Symmetry analysis for {pairing} pairing, impurity {imp_tag}\n')
                report.write(f'Parameters: mu={mu}, alpha={alpha}, delta_mag={delta_mag}, gamma={gamma}, gap_size={gap_size}\n')
                report.write(f'omega={om}, v_mag={v_mag}, q0_factor={q0_factor}, Nk={Nk}, Nq={Nq}\n\n')
                report.write('C6v irrep projection (weight = sum|f_nu|^2 / sum|f|^2):\n')
                report.write(f'Datasets with max |f| < {zero_tol:.2e} are numerically zero\n')

                datasets = []
                for m, ch in enumerate(CHANNELS):
                    sub = rf'_{ch}' if ch != '0' else ''
                    for name, label, img, kind, U in [
                        (f'Re drho_{ch}(q)', rf'$\mathrm{{Re}}\,\delta\rho{sub}(\bvec{{q}})$', np.real(drho[m]), 'q', Uq),
                        (f'Im drho_{ch}(q)', rf'$\mathrm{{Im}}\,\delta\rho{sub}(\bvec{{q}})$', np.imag(drho[m]), 'q', Uq),
                        (f'drho_{ch}(r)', rf'$\delta\rho{sub}(\bvec{{r}})$', np.real(drho_r[m]), 'r', Ur),
                    ]:
                        fnu, weights, maxvals = irrep_decomposition(img, geom[kind][0], U)
                        negligible = np.amax(np.abs(img)) < zero_tol
                        if negligible:
                            weights = np.zeros_like(weights)
                        datasets.append(dict(name=name, label=label, img=img, kind=kind,
                                             fnu=fnu, weights=weights, maxvals=maxvals, negligible=negligible))
                    report.write(f'\nChannel sigma_{ch}:\n')
                    for d in datasets[-3:]:
                        write_irrep_analysis(report, d['name'], d['weights'], d['maxvals'], d['negligible'])
                    report.write(f'Check: max |Im drho_{ch}(r)| = {np.amax(np.abs(np.imag(drho_r[m]))):.4e}\n')

            plot_drho(datasets, geom, r_window, os.path.join(out_dir, f'drho_{tag}.png'), title)
            plot_irreps(datasets, geom, r_window, os.path.join(out_dir, f'irreps_{tag}.png'), title)
            plot_note(datasets, geom, r_window, os.path.join(out_dir, f'note_drho_{tag}.pdf'))
            np.savez(os.path.join(out_dir, f'drho_{tag}.npz'), drho=drho, drho_r=drho_r, zero_tol=zero_tol,
                     omega=om, gap_size=gap_size)


if __name__ == '__main__':
    pairings = ['s', 'ext s', 'd+id', 'dx2y2']
    params = dict(delta_mag=1e-1, gamma=1e-3, Nk=240, Nq=120)
    for pairing in pairings:
        main(pairing, impurity='A1', q0_factors=(1.0,), **params)
    for pairing in pairings:
        main(pairing, impurity='A2', q0_factors=(20.0,), **params)
