NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

%.html: %.ipynb
	jupyter nbconvert $< --to html

install:
	conda env create -f environment.yml

readme:
	grip README.md

html:
	grip --export README.md
	jupyter nbconvert $(NB) --to html

.PHONY: run $(NB) clean install readme html
