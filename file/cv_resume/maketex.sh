mainname=$1
me=_lding
xelatex ${mainname}${me}.tex
bibtex ${mainname}${me}
xelatex ${mainname}${me}.tex
xelatex ${mainname}${me}.tex

rm  *.aux *.log  *.blg  mainNotes.bib

open ${me}${mainname}.pdf
