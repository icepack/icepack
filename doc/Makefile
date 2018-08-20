# This environment variable needs to be set in order to find the data sets used
# for some of the demos.
check-env:
ifndef ICEPACK_DATA
    $(error ICEPACK_DATA is undefined)
endif

source/index.rst:
	python3 make_index.py $(DEMO_FILES)

DEMO_FILES:=$(shell find ../demo -not -path '*/\.*' -type f -name '*\.ipynb' | sort)
notebooks: $(patsubst ../demo/%.ipynb,source/icepack.demo.%.rst,$(DEMO_FILES))

# For each demo, execute it with a 10-min. timeout, convert it to restructured
# text, and save the result in the `source` directory with `icepack.demo.`
# prepended to the name.
source/icepack.demo.%.rst: ../demo/%.ipynb
	PYTHONPATH=`dirname $<` jupyter nbconvert \
			   --to rst \
			   --execute \
			   --ExecutePreprocessor.timeout=24000 \
			   --ExecutePreprocessor.kernel_name="firedrake" \
			   --output-dir=./source \
			   --output=`basename $@` \
			   $<

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = icepack
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf build source/icepack* source/index.rst source/modules.rst

.PHONY: help clean Makefile

html: Makefile notebooks source/index.rst
	sphinx-apidoc --force --separate --maxdepth=1 --module-first -o source ../icepack
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
