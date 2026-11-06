#!/bin/sh
# Copy the resonant-case table into the breaking_TRS note. Run from this folder.
set -e
NOTE=../notes/2026-11-03_breaking_TRS
mkdir -p $NOTE/figs $NOTE/tables
python resonant_table.py > /dev/null
cp resonant_table.tex $NOTE/tables/fullT_resonant.tex
