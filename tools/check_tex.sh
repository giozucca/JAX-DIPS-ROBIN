#!/bin/sh
# Find undefined references, unused labels, and duplicate labels in a LaTeX file.
#   sh tools/check_tex.sh main.tex
F="$1"
echo "=== UNDEFINED REFERENCES  (these render as ??) ==="
grep -o '\\\(eq\)\?ref{[^}]*}' "$F" | sed 's/.*{//;s/}//' | sort -u > /tmp/_refs
grep -o '\\label{[^}]*}'       "$F" | sed 's/.*{//;s/}//' | sort -u > /tmp/_labs
comm -23 /tmp/_refs /tmp/_labs | sed 's/^/  MISSING LABEL: /'
echo
echo "=== DUPLICATE LABELS  (multiply-defined) ==="
grep -o '\\label{[^}]*}' "$F" | sed 's/.*{//;s/}//' | sort | uniq -d | sed 's/^/  DUPLICATE: /'
echo
echo "=== LABELS NEVER REFERENCED  (harmless, but figures/tables should be cited) ==="
comm -13 /tmp/_refs /tmp/_labs | sed 's/^/  unused: /'
echo
echo "=== LEFTOVER MARKERS ==="
grep -n 'TODO\|placeholder\|myFigure\|star_convergence_2' "$F" | sed 's/^/  /'
echo
echo "=== MISSING PACKAGE ==="
grep -q 'usepackage{float}' "$F" || echo "  \\usepackage{float} is MISSING but [H] is used"
