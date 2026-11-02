PYTHON ?= python

.PHONY: all fetch dataset models models-fast emotions arc notebook clean palette report

all: dataset models emotions arc notebook

fetch:            ## download the three screenplays into data/raw/
	$(PYTHON) scripts/00_fetch_scripts.py

dataset:          ## parse the screenplays into one tidy dialogue table
	$(PYTHON) scripts/01_build_dataset.py

models:           ## train + validate the faction classifiers (random vs grouped split)
	$(PYTHON) scripts/02_train_faction_models.py

models-fast:      ## same, TF-IDF only, no transformer
	$(PYTHON) scripts/02_train_faction_models.py --skip-transformer

emotions:         ## score every line with the external emotion model
	$(PYTHON) scripts/03_score_emotions.py

arc:              ## statistics + figures
	$(PYTHON) scripts/04_analyze_arc.py

notebook:         ## rebuild and execute the narrative notebook
	$(PYTHON) scripts/05_build_notebook.py
	cd notebooks && PYTHONPATH=../src $(PYTHON) -m nbconvert --to notebook --execute --inplace anakin_fall.ipynb

palette:          ## re-run the five colour checks on the report palette
	PYTHONPATH=src $(PYTHON) -m anakin_nlp.palette

report:           ## recompile the project report (docs/project_report.pdf)
	cd docs && xelatex -interaction=nonstopmode project_report.tex >/dev/null \
	        && xelatex -interaction=nonstopmode project_report.tex >/dev/null \
	        && rm -f *.aux *.log *.out
	@echo "-> docs/project_report.pdf"

clean:
	rm -rf data/processed/* figures/* reports/* __pycache__ src/anakin_nlp/__pycache__
