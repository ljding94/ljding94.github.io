mainname=$1
#me=Lijie_Ding_
xelatex ${me}${mainname}.tex
bibtex ${me}${mainname}
xelatex ${me}${mainname}.tex
xelatex ${me}${mainname}.tex

rm  *.aux *.log  *.blg  mainNotes.bib

open ${me}${mainname}.pdf
