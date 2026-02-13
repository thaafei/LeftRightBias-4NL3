# Makefile
# From https://danielkaes.wordpress.com/2009/03/14/compiling-latex-documents-using-makefiles/

TEX=pdflatex
BIBTEX=bibtex
BUILDTEX=$(TEX) step_1.tex
BUILDBIB=$(BIBTEX) step_1


all:
	${BUILDTEX}
	${BUILDBIB}
	${BUILDTEX}
	${BUILDTEX}



clean-all:
	rm -f *.dvi *.log *.bak *.aux *.bbl *.blg *.idx *.ps *.eps *.pdf *.toc *.out *~

clean:
	rm -f *.log *.bak *.aux *.bbl *.blg *.idx *.toc *.out *~