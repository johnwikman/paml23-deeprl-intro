.PHONY: all handout paused clean

SHELL = bash

BUILD-DIR = _build
PDFLATEX = pdflatex -shell-escape -output-directory=$(BUILD-DIR)

all: handout

handout:
	make deeprl-intro.pdf docclass-args=handout

paused:
	make deeprl-intro.pdf docclass-args=

deeprl-intro.pdf:
	$(eval TEXFILE := $(BUILD-DIR)/deeprl-intro.tex)
	$(eval PDFFILE := $(BUILD-DIR)/deeprl-intro.pdf)
	mkdir -p $(BUILD-DIR)
	mkdir -p $(BUILD-DIR)/img
	m4 -D ADDITIONAL_DOCCLASS_ARGS=$(docclass-args) deeprl-intro.tex > $(TEXFILE)
	cp -r img/* $(BUILD-DIR)/img
	$(PDFLATEX) $(TEXFILE)
	$(PDFLATEX) $(TEXFILE)
	$(PDFLATEX) $(TEXFILE)
	mv $(PDFFILE) .

clean:
	rm -rf $(BUILD-DIR)
