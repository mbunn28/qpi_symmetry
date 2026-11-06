#!/bin/sh
# Regenerate the Rashba figures and copy them into the breaking_TRS note. Run from this folder.
set -e
NOTE=../notes/2026-11-03_breaking_TRS
mkdir -p $NOTE/figs
python combined_bands.py
python rashba_spin.py
cp figs/bands_combined.pdf figs/dpid_compare.pdf figs/rashba_born_nonchiral.pdf figs/frg_compare.pdf figs/frg_nonchiral.pdf $NOTE/figs/
