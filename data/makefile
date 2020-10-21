NB = $(sort $(wildcard *.ipynb))

run: $(NB)

$(NB):
	jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 $@

clean:
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

%.html: %.ipynb
	jupyter nbconvert $< --to html

html:
	jupyter nbconvert $(NB) --to html

download:
	wget -i s2_raw/manifest.txt -B https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/ -P s2_raw/

preprocess:
	python s2_1_corpus_to_bipartite.py
	python s2_2_downsample_bipartite.py
	python s2_3_bipartite_to_graphs.py
	python s2_3_bipartite_to_complex.py
	python s2_4_complex_to_matrices.py

.PHONY: run $(NB) clean html download preprocess
