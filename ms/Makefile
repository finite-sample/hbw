.PHONY: all paper figures clean

all: figures paper

paper: paper.pdf

paper.pdf: paper.tex paper.bib figures
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

figures: figures/kde_ise_comparison.pdf figures/nw_mse_comparison.pdf figures/eval_count_comparison.pdf figures/convergence_trace.pdf

figures/kde_ise_comparison.pdf figures/nw_mse_comparison.pdf figures/eval_count_comparison.pdf figures/convergence_trace.pdf: generate_figures.py
	cd .. && uv run python ms/generate_figures.py

clean:
	rm -f paper.aux paper.bbl paper.blg paper.log paper.out paper.pdf
	rm -f figures/*.pdf
