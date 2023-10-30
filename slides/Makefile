.PHONY: all clean

SHELL = bash

BUILD-DIR = _build
PDFLATEX = pdflatex -shell-escape -output-directory=$(BUILD-DIR)

all: deeprl-intro.pdf

deeprl-intro.pdf:
	$(eval TEXFILE := $(BUILD-DIR)/deeprl-intro.tex)
	$(eval PDFFILE := $(BUILD-DIR)/deeprl-intro.pdf)
	mkdir -p $(BUILD-DIR)
	mkdir -p $(BUILD-DIR)/img
	cp deeprl-intro.tex $(TEXFILE)
	cp -r img/* $(BUILD-DIR)/img
	$(PDFLATEX) $(TEXFILE)
	$(PDFLATEX) $(TEXFILE)
	$(PDFLATEX) $(TEXFILE)
	mv $(PDFFILE) .

clean:
	rm -rf $(BUILD-DIR)
