#!/bin/sh
# Copy the full-T figures (no q cutoff, data_nocut/) and the resonant-case table into the breaking_TRS note. Run from this folder.
set -e
NOTE=../notes/2026-11-03_breaking_TRS
D=data_nocut/delta0.1_gamma0.001_alpha0.1/dpid
mkdir -p $NOTE/figs $NOTE/tables
cp $D/full/note_drho_om0.0000_V0.1000_q0inf.pdf  $NOTE/figs/fullT_rashba_dpid_om0.pdf
cp $D/res0/note_drho_om0.0000_V6.2799_q0inf.pdf  $NOTE/figs/fullT_rashba_dpid_om0_res.pdf
cp "data_nocut/frg_nu0.5_U8_a0.4_nk24_nkf31/delta_rms0.1_gamma0.001/frg_chi+1/full/note_drho_om0.0846_V0.1000_q0inf.pdf" $NOTE/figs/fullT_frg_chiral.pdf
python resonant_table.py > /dev/null
cp resonant_table.tex $NOTE/tables/fullT_resonant.tex
