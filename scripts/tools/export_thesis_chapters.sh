#!/usr/bin/env bash
# Build full thesis PDF and one PDF per chapter/frontmatter unit.
# Always runs inside TeX Live Docker (Factory6G AGENTS.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THESIS="${ROOT}/thesis"
OUT="${THESIS}/exports"
IMAGE="${THESIS_TEX_IMAGE:-texlive/texlive:latest}"

mkdir -p "${OUT}"

run_latex() {
  local texfile="$1"
  docker run --rm \
    -v "${THESIS}:/work" \
    -w /work \
    "${IMAGE}" \
    latexmk -g -synctex=1 -interaction=nonstopmode -file-line-error -xelatex "${texfile}"
}

echo "==> Building full thesis (main.tex)"
run_latex main.tex
cp -f "${THESIS}/main.pdf" "${OUT}/thesis_full.pdf"
echo "    -> ${OUT}/thesis_full.pdf"

# Standalone chapter wrappers live under thesis/exports_src/
WRAPPER_DIR="${THESIS}/exports_src"
mkdir -p "${WRAPPER_DIR}"

chapters=(
  "00_abstract:abstract"
  "01_introduction:ch01_introduction"
  "02_literature_review:ch02_literature_review"
  "03_methodology:ch03_methodology"
  "04_results:ch04_results"
  "05_discussion:ch05_discussion"
  "06_conclusions:ch06_conclusions"
)

for entry in "${chapters[@]}"; do
  stem="${entry%%:*}"
  target="${entry##*:}"
  wrapper="${WRAPPER_DIR}/${stem}.tex"

  if [[ "${target}" == "abstract" ]]; then
    body='\input{frontmatter/abstract}'
  else
    body="\\input{chapters/${target}}"
  fi

  cat > "${wrapper}" <<EOF
% Auto-generated standalone export — do not edit by hand.
% Rebuild via: scripts/tools/export_thesis_chapters.sh
\\documentclass[12pt,a4paper,oneside]{report}

\\usepackage[a4paper,left=4cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\\usepackage{fontspec}
\\IfFontExistsTF{Times New Roman}{%
  \\setmainfont{Times New Roman}%
}{%
  \\setmainfont{TeX Gyre Termes}%
}
\\usepackage{setspace}
\\onehalfspacing
\\usepackage{graphicx}
\\usepackage{amsmath,amssymb}
\\usepackage{array}
\\usepackage{tabularx}
\\usepackage{booktabs}
\\usepackage{titlesec}
\\usepackage{algorithm}
\\usepackage{algpseudocode}
\\usepackage{microtype}
\\usepackage[hidelinks]{hyperref}
\\usepackage{xurl}
\\urlstyle{tt}
\\newcommand{\\tabpath}[1]{\\nolinkurl{#1}}
\\usepackage{cleveref}
\\usepackage[backend=biber,style=ieee,maxbibnames=10]{biblatex}
\\addbibresource{references.bib}
\\DeclareNameAlias{sortname}{giveninits}

\\titleformat{\\chapter}[display]
  {\\normalfont\\bfseries\\centering}
  {CHAPTER \\thechapter}
  {0.5\\baselineskip}
  {\\bfseries\\MakeUppercase}
\\titlespacing*{\\chapter}{0pt}{0pt}{20pt}
\\titleformat{\\section}{\\normalfont\\bfseries}{\\thesection}{1em}{}
\\titleformat{\\subsection}{\\normalfont\\bfseries}{\\thesubsection}{1em}{}

\\setcounter{secnumdepth}{3}
\\graphicspath{{figures/}{../figures/}}

\\begin{document}
\\pagenumbering{arabic}
${body}
\\cleardoublepage
\\printbibliography[title={REFERENCES}]
\\end{document}
EOF

  echo "==> Building ${stem}.pdf"
  # Compile from thesis root so relative paths resolve; wrapper is under exports_src/
  docker run --rm \
    -v "${THESIS}:/work" \
    -w /work \
    "${IMAGE}" \
    latexmk -g -synctex=1 -interaction=nonstopmode -file-line-error -xelatex \
      -outdir=exports_src "exports_src/${stem}.tex"
  cp -f "${WRAPPER_DIR}/${stem}.pdf" "${OUT}/${stem}.pdf"
  echo "    -> ${OUT}/${stem}.pdf"
done

echo ""
echo "Exports ready in ${OUT}:"
ls -la "${OUT}"
