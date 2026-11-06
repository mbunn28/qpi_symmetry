#!/bin/sh
# Regenerate the case table and copy it and the combined no-SOC figures into the breaking_TRS note. Run from this folder.
set -e
NOTE=../notes/2026-11-03_breaking_TRS
mkdir -p $NOTE/figs $NOTE/tables
python case_table.py > /dev/null
cp case_table.tex $NOTE/tables/cases.tex
cp figs/nosoc_A1imp_all.pdf figs/nosoc_A2imp_nonzero.pdf $NOTE/figs/
